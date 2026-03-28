from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from importlib import resources

from accelerate.utils import broadcast_object_list
from tqdm import tqdm

from conti.config_schema import ContiConfig
from conti.data.loaders import load_math_prompts
from conti.eval.drift import DriftTracker
from conti.eval.metrics import eval_math_pass1, eval_safety_asr_proxy, eval_safety_multi_benchmark
from conti.logging import ExperimentLogger
from conti.replay.buffer import SafetyReplayBuffer
from conti.reproducibility import RunManifest, build_manifest, set_global_seed
from conti.training.format import build_supervised_example, replay_item_to_text, user_prompt_only
from conti.training.sft import SFTTrainer
from conti.verifier.keyword import KeywordVerifier, CompositeVerifier


def _broadcast_texts(accelerator, texts: list[str] | None) -> list[str]:
    if accelerator.num_processes <= 1:
        return texts or []
    payload = [texts if accelerator.is_main_process else None]
    broadcast_object_list(payload, from_process=0)
    return payload[0] or []


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _extract_num(text: str) -> str | None:
    nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return nums[-1] if nums else None


def _extract_gold_num(gold: str | None) -> str | None:
    if not gold:
        return None
    m = re.search(r"####\s*([-+]?[\d,]+)", gold)
    if m:
        return m.group(1).replace(",", "")
    return _extract_num(gold)


def _is_correct(generated: str, gold: str | None) -> bool:
    g = _extract_gold_num(gold)
    p = _extract_num(generated)
    if g is None or p is None:
        return False
    return p.replace(".", "") == g.replace(".", "")


def _validate_protocol(cfg: ContiConfig) -> None:
    exp = cfg.loop.experiment
    if exp == "baseline_single_sft" and cfg.loop.num_self_improve_rounds != 1:
        raise ValueError("baseline_single_sft must run exactly 1 round")  # ek hi round chalega bhai
    if exp != "phase2_verifier_buffer" and cfg.replay.enabled:
        raise ValueError("replay buffer only for phase2_verifier_buffer")
    if exp == "phase2_verifier_buffer" and not cfg.replay.enabled:
        raise ValueError("phase2_verifier_buffer needs replay.enabled=true")


@torch.inference_mode()
def _gen_batch(model, tokenizer, questions, max_new_tokens, device, temp, top_p):
    texts = [user_prompt_only(tokenizer, q) for q in questions]
    max_in = min(2048, getattr(tokenizer, "model_max_length", 2048) or 2048)
    enc = tokenizer(texts, padding=True, return_tensors="pt", truncation=True, max_length=max_in)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens, do_sample=True,
        top_p=top_p, temperature=temp,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = []
    for i, row in enumerate(out):
        inp_len = int(enc["attention_mask"][i].sum())
        gen.append(tokenizer.decode(row[inp_len:], skip_special_tokens=True).strip())
    return gen


def _make_verifier(cfg: ContiConfig):
    if cfg.verifier.mode == "keyword_and_model":
        return CompositeVerifier(cfg.verifier)
    return KeywordVerifier(cfg.verifier)


def _run_evals(cfg, model, tokenizer, device):
    math = eval_math_pass1(
        model, tokenizer, device,
        dataset_name=cfg.data.math_dataset,
        split=cfg.data.math_test_split,
        max_samples=min(200, cfg.data.max_prompts_per_eval * 2),
    )
    safety = eval_safety_multi_benchmark(
        model, tokenizer, device,
        benchmark_names=cfg.data.safety_eval_datasets,
        max_per_benchmark=cfg.data.max_prompts_per_eval,
    )
    return {"math_pass1": math, "safety_asr": safety}


def _get_asr_scores(safety_results: dict) -> dict[str, float]:
    return {
        bench: r["asr_proxy"]
        for bench, r in safety_results.items()
        if isinstance(r, dict) and "asr_proxy" in r
    }


def run_experiment(cfg: ContiConfig) -> None:
    _validate_protocol(cfg)
    set_global_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: RunManifest = build_manifest(cfg.to_dict(), cfg.seed)
    manifest.save(out / "run_manifest.json")

    from accelerate import Accelerator

    acc = Accelerator(gradient_accumulation_steps=cfg.loop.gradient_accumulation_steps)
    trainer = SFTTrainer(cfg=cfg, accelerator=acc)
    logger = ExperimentLogger(cfg, run_dir=out)
    drift = DriftTracker()  # har round ke baad check karenge kitna safety gira

    # baseline_frozen: evaluate only
    if cfg.loop.experiment == "baseline_frozen":
        model, tokenizer = trainer.load_model_tokenizer()
        model = trainer.ensure_model_prepared(model)
        model.eval()
        if acc.is_main_process:
            metrics = _run_evals(cfg, model, tokenizer, acc.device)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            drift.set_baseline(_get_asr_scores(metrics.get("safety_asr", {})))
        logger.finish()  # bas evaluate karo, kuch train mat karo

        return

    model, tokenizer = trainer.load_model_tokenizer()
    verifier = _make_verifier(cfg)

    replay_path = cfg.replay.dataset_path
    if replay_path is None:
        replay_path = resources.files("conti.data") / "safety_replay.jsonl"
    replay_buf = SafetyReplayBuffer(replay_path, cfg.seed)
    if not cfg.replay.enabled:
        replay_buf = SafetyReplayBuffer(None, cfg.seed)

    # ewc setup if needed
    if cfg.ewc.enabled:
        from conti.training.ewc import EWCPenalty
        ewc = EWCPenalty(model, lambda_ewc=cfg.ewc.lambda_ewc)
        # TODO: actually compute fisher on safety data.. need dataloader for that
        trainer.set_ewc(ewc)

    math_items = load_math_prompts(
        cfg.data.math_dataset, cfg.data.math_split,
        max_samples=cfg.loop.max_train_samples_per_round,
    )

    # get baseline scores before we touch anything
    if acc.is_main_process:
        m = trainer.ensure_model_prepared(model)
        m.eval()
        base_metrics = _run_evals(cfg, m, tokenizer, acc.device)
        drift.set_baseline(_get_asr_scores(base_metrics.get("safety_asr", {})))
        logger.log_round(-1, {"baseline": base_metrics})

    filt_log: list[dict[str, Any]] = []
    rnd_metrics: list[dict[str, Any]] = []
    sup_rounds: list[list[str]] = []

    for ri in range(cfg.loop.num_self_improve_rounds):
        if cfg.loop.experiment == "baseline_single_sft" and ri > 0:
            break

        rnd_texts: list[str] = []
        bs = cfg.loop.generate_batch_size
        rej_correct = 0

        if cfg.loop.experiment == "baseline_single_sft":
            if acc.is_main_process:
                rnd_texts = _make_static_sft(tokenizer, math_items)
        else:
            mdl = trainer.ensure_model_prepared(model)
            dev = acc.device
            mdl.eval()

            pend_q: list[str] = []
            pend_m: list[dict[str, Any]] = []

            def flush():
                nonlocal rnd_texts, pend_q, pend_m, rej_correct
                if not pend_q or not acc.is_main_process:
                    return
                qs, meta = pend_q, pend_m
                pend_q, pend_m = [], []

                gens = _gen_batch(
                    mdl, tokenizer, qs,
                    max_new_tokens=min(cfg.loop.generation_max_new_tokens, cfg.model.max_seq_length // 2),
                    device=dev, temp=cfg.loop.generation_temperature,
                    top_p=cfg.loop.generation_top_p,
                )
                for q, g, m in zip(qs, gens, meta, strict=True):
                    if cfg.loop.require_correct_trajectory:
                        if not _is_correct(g, m.get("gold_answer")):
                            rej_correct += 1
                            continue

                    use_v = cfg.loop.experiment in ("phase1_verifier", "phase2_verifier_buffer")
                    if use_v:
                        v = verifier.check_text(g)
                        filt_log.append({
                            "round": ri, "id": m["id"],
                            "safe": v.safe, "reason": v.reason,
                        })
                        if not v.safe:
                            continue

                    rnd_texts.append(build_supervised_example(tokenizer, q, g))

            it = tqdm(math_items, desc=f"gen r{ri}") if acc.is_main_process else math_items
            for item in it:
                if acc.is_main_process:
                    pend_q.append(item["question"])
                    pend_m.append({"id": item["id"], "gold_answer": item.get("answer")})
                    if len(pend_q) >= bs:
                        flush()
            if acc.is_main_process:
                flush()

            # phase2: mix in replay buffer
            if cfg.loop.experiment == "phase2_verifier_buffer" and cfg.replay.enabled and len(replay_buf):
                n = min(
                    cfg.replay.safety_samples_per_batch * max(1, len(rnd_texts) // max(1, bs)),
                    len(replay_buf),
                )
                if acc.is_main_process:
                    for ex in replay_buf.sample(n):
                        try:
                            rnd_texts.append(replay_item_to_text(tokenizer, ex))
                        except (ValueError, KeyError, TypeError):
                            continue

        rnd_texts = _broadcast_texts(acc, rnd_texts)
        sup_rounds.append(rnd_texts)
        accepted = len(rnd_texts)

        rej_safety = 0
        if cfg.loop.experiment in ("phase1_verifier", "phase2_verifier_buffer"):
            rej_safety = sum(1 for x in filt_log if x["round"] == ri and not x["safe"])

        # train
        if cfg.loop.experiment in ("naive_continual", "phase1_verifier", "phase2_verifier_buffer", "baseline_single_sft"):
            stats = trainer.train_on_texts(
                model=model, tokenizer=tokenizer, texts=rnd_texts,
                learning_rate=cfg.loop.learning_rate,
                num_epochs=cfg.loop.num_epochs_per_round,
                micro_batch_size=cfg.loop.train_micro_batch_size,
                warmup_ratio=cfg.loop.warmup_ratio,
                weight_decay=cfg.loop.weight_decay,
                max_seq_length=cfg.model.max_seq_length,
                max_grad_norm=cfg.loop.max_grad_norm,
            )

            ckpt = out / "checkpoints" / f"round_{ri}"
            trainer.save_pretrained(trainer.ensure_model_prepared(model), tokenizer, str(ckpt))

            if acc.is_main_process and stats:
                for si, (loss, ewc_p) in enumerate(zip(stats.get("loss", []), stats.get("ewc_penalty", []))):
                    logger.log_step({"loss": loss, "ewc_penalty": ewc_p}, step=ri * 10000 + si)

        # eval
        if cfg.loop.eval_every_round and acc.is_main_process:
            model.eval()
            ev = _run_evals(cfg, trainer.ensure_model_prepared(model), tokenizer, acc.device)
            ev["round"] = ri
            ev["accepted"] = accepted
            ev["rej_verifier"] = rej_safety
            ev["rej_correctness"] = rej_correct

            cur_asrs = _get_asr_scores(ev.get("safety_asr", {}))
            drift.record(ri, cur_asrs)
            ev["drift"] = drift.get_round_summary(ri)

            rnd_metrics.append(ev)
            logger.log_round(ri, ev)
            logger.log_drift(ri, ev.get("drift", {}))

        acc.wait_for_everyone()

    # save everything
    if acc.is_main_process:
        _write_jsonl(out / "logs" / "filter_log.jsonl", filt_log)
        _write_jsonl(out / "logs" / "round_metrics.jsonl", rnd_metrics)
        (out / "supervised_counts.json").write_text(
            json.dumps([len(x) for x in sup_rounds]), encoding="utf-8",
        )
        drift.save(out / "logs" / "drift_log.jsonl")

        model.eval()
        final = _run_evals(cfg, trainer.ensure_model_prepared(model), tokenizer, acc.device)
        final["drift_summary"] = drift.to_summary_dict()
        (out / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    logger.finish()


def _make_static_sft(tokenizer, math_items):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    texts = []
    for i, row in enumerate(ds):
        if i >= len(math_items):
            break
        texts.append(build_supervised_example(tokenizer, row["question"].strip(), row["answer"].strip()))
    return texts
