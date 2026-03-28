from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm


class EWCPenalty:
    def __init__(self, model: nn.Module, lambda_ewc: float = 5000.0):
        self.lambda_ewc = lambda_ewc
        self._ref_params: dict[str, torch.Tensor] = {}
        self._fisher_diag: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self._ref_params[name] = param.data.clone().detach()

    def compute_fisher(
        self,
        model: nn.Module,
        dataloader: Any,
        device: torch.device,
        max_samples: int = 200,
    ) -> None:
        model.eval()
        fisher_accum: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_accum[name] = torch.zeros_like(param.data)

        n_samples = 0
        for batch in tqdm(dataloader, desc="fisher", leave=False):
            if n_samples >= max_samples:
                break
            model.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            if "input_ids" not in inputs:
                continue
            outputs = model(**inputs, labels=inputs.get("input_ids"))
            loss = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accum[name] += param.grad.data ** 2

            n_samples += inputs["input_ids"].shape[0]

        for name in fisher_accum:
            fisher_accum[name] /= max(n_samples, 1)

        self._fisher_diag = fisher_accum
        model.zero_grad()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self._fisher_diag and name in self._ref_params:
                fisher = self._fisher_diag[name].to(param.device)
                ref = self._ref_params[name].to(param.device)
                total = total + (fisher * (param - ref) ** 2).sum()
        return self.lambda_ewc * total

    @property
    def is_ready(self) -> bool:
        return len(self._fisher_diag) > 0
