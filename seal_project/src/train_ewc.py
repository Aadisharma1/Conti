"""
SFT training with Elastic Weight Consolidation (EWC).

Standard bfloat16 LoRA fine-tuning with a custom EWC penalty
to prevent catastrophic forgetting of safety alignment.

No QLoRA, no bitsandbytes. Pure bfloat16 on A100 40GB.
"""

import os
import json
import copy
import argparse
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ─── Dataset ──────────────────────────────────────────────────

class SelfEditDataset(Dataset):
    """Loads JSONL self-edit trajectories for SFT."""

    def __init__(self, path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"  loaded {len(self.samples)} samples from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        instruction = item.get("instruction", "")
        response = item.get("response", item.get("self_edit", ""))

        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        # mask padding tokens in labels
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ─── EWC Implementation ──────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation.

    Computes Fisher Information Matrix (diagonal) from the model's
    current parameter state, then penalizes deviations during new training.
    This preserves the base model's safety alignment while learning new tasks.
    """

    def __init__(self, model, dataloader, device, n_samples: int = 200):
        self.params = {}     # theta_star (old params)
        self.fisher = {}     # diagonal Fisher

        self._compute_fisher(model, dataloader, device, n_samples)

    def _compute_fisher(self, model, dataloader, device, n_samples: int):
        """Compute diagonal Fisher from model gradients on a reference dataset."""
        print("  computing Fisher Information Matrix...")
        model.eval()

        # store current params
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()
                self.fisher[name] = torch.zeros_like(param.data)

        count = 0
        for batch in tqdm(dataloader, desc="  fisher", total=min(n_samples, len(dataloader))):
            if count >= n_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2

            count += 1

        # normalize
        for name in self.fisher:
            self.fisher[name] /= max(count, 1)

        model.zero_grad()
        print(f"  Fisher computed over {count} batches")

    def penalty(self, model) -> torch.Tensor:
        """EWC penalty: sum_i F_i * (theta_i - theta_star_i)^2"""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher:
                fisher = self.fisher[name].to(param.device)
                old_param = self.params[name].to(param.device)
                loss += (fisher * (param - old_param) ** 2).sum()
        return loss


# ─── Training Loop ────────────────────────────────────────────

def train(
    model,
    tokenizer,
    train_dataset: SelfEditDataset,
    fisher_dataset: Optional[SelfEditDataset],
    args,
):
    device = next(model.parameters()).device

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # compute EWC if lambda > 0
    ewc = None
    if args.ewc_lambda > 0 and fisher_dataset is not None:
        fisher_loader = DataLoader(
            fisher_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        ewc = EWC(model, fisher_loader, device, n_samples=args.fisher_samples)
    elif args.ewc_lambda > 0:
        # use training data itself for Fisher (less ideal but works)
        ewc = EWC(model, train_loader, device, n_samples=args.fisher_samples)

    # optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    print(f"\n  training for {args.epochs} epochs ({total_steps} steps)")
    print(f"  ewc_lambda: {args.ewc_lambda}")
    print(f"  lr: {args.lr}")
    print(f"  batch_size: {args.batch_size}")

    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss, epoch_ewc_loss = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"  epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            sft_loss = outputs.loss

            # EWC penalty
            ewc_loss = torch.tensor(0.0, device=device)
            if ewc is not None:
                ewc_loss = args.ewc_lambda * ewc.penalty(model)

            total_loss = sft_loss + ewc_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += sft_loss.item()
            epoch_ewc_loss += ewc_loss.item()
            global_step += 1

            pbar.set_postfix({
                "sft": f"{sft_loss.item():.4f}",
                "ewc": f"{ewc_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = epoch_loss / len(train_loader)
        avg_ewc = epoch_ewc_loss / len(train_loader)
        print(f"  epoch {epoch+1}: sft_loss={avg_loss:.4f}, ewc_loss={avg_ewc:.4f}")

    return model


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT + EWC training")
    parser.add_argument("--base-model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--train-data", type=str, required=True,
                        help="path to JSONL training data")
    parser.add_argument("--fisher-data", type=str, default=None,
                        help="path to JSONL data for Fisher computation (safety data)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ewc_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--ewc-lambda", type=float, default=0.5,
                        help="EWC penalty weight. 0 = no EWC (naive SFT)")
    parser.add_argument("--fisher-samples", type=int, default=200,
                        help="number of batches for Fisher computation")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    # load model in bfloat16
    print(f"loading {args.base_model} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # wrap in LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # load datasets
    train_dataset = SelfEditDataset(args.train_data, tokenizer, args.max_length)

    fisher_dataset = None
    if args.fisher_data:
        fisher_dataset = SelfEditDataset(args.fisher_data, tokenizer, args.max_length)

    # train
    model = train(model, tokenizer, train_dataset, fisher_dataset, args)

    # save
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"\n  model saved to {out_path}")


if __name__ == "__main__":
    main()
