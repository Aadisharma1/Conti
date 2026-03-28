# conti-safety

Continual learning safety framework for self-improving LLMs. Built to experimentally measure and fix the safety drift that happens when models recursively train on their own synthetic data.

## what this does

Self-improving LLMs (like STaR/SEAL) keep getting better at reasoning tasks by generating and training on their own solutions. But nobody really checked what happens to the safety guardrails during this process. Turns out — they degrade. Badly.

This repo:
1. **Measures the safety drift** — tracks how ASR (attack success rate) changes across self-improvement rounds
2. **Fixes it** — plugs a lightweight safety verifier into the self-improvement loop to filter unsafe trajectories before SFT
3. **Anchors safety weights** — optionally uses EWC (Elastic Weight Consolidation) + a safety replay buffer to prevent catastrophic forgetting of RLHF alignment

## experiment arms (from the research proposal)

| Arm | What it does |
|-----|-------------|
| `baseline_frozen` | Just evaluate the seed model, no training |
| `baseline_single_sft` | One-shot SFT on static human-verified math data |
| `naive_continual` | Recursive self-improvement with NO safety filters |
| `phase1_verifier` | Self-improvement + safety verifier filtering |
| `phase2_verifier_buffer` | Verifier + safety replay buffer injection |

## install

```bash
pip install -e .
```

Needs Python 3.10+. For Llama models, set `HF_TOKEN` in your environment.

## quick run (single GPU)

```bash
python scripts/run_experiment.py --config configs/default.yaml
```

## multi-GPU (single node)

```bash
accelerate launch --multi_gpu scripts/run_experiment.py --config configs/default.yaml
```

## full sweep (all arms × multiple seeds)

```bash
python scripts/run_sweep.py \
  --base-config configs/default.yaml \
  --out-root outputs/sweep_v1 \
  --experiments baseline_frozen baseline_single_sft naive_continual phase1_verifier phase2_verifier_buffer \
  --seeds 11 22 33

# aggregate results
python scripts/aggregate_results.py \
  --root outputs/sweep_v1 \
  --experiments baseline_frozen baseline_single_sft naive_continual phase1_verifier phase2_verifier_buffer \
  --out outputs/sweep_v1/aggregate.json

# generate plots
python scripts/plot_results.py \
  --sweep-root outputs/sweep_v1 \
  --aggregate outputs/sweep_v1/aggregate.json
```

## safety benchmarks

We evaluate on 3 benchmarks:
- **AdvBench** — harmful instruction following (direct attacks)
- **XSTEST** — ambiguous prompts that test over-refusal
- **do-not-answer** — broad coverage harmful queries

## key features

- **Verifier modes**: keyword-only (fast, no setup) or keyword + model-based classifier (more accurate, needs a small model)
- **EWC regularization**: anchors safety-critical weights using Fisher Information
- **Drift tracking**: per-round safety drift measurement with JSONL logs
- **wandb integration**: optional, just set `WANDB_API_KEY`
- **Multi-GPU**: via HF Accelerate, works on single node or multi-node clusters
- **Reproducibility**: every run dumps a manifest with config hash, git state, package versions, seeds

## wandb setup (optional)

```bash
export WANDB_API_KEY=your_key_here
```

Then set `logging.use_wandb: true` in your config. Metrics, drift curves, and training losses will appear in your wandb dashboard.

## smoke test (no GPU needed)

```bash
python scripts/run_experiment.py --config configs/smoke_gpt2.yaml
```

Uses GPT-2 so no Llama weights required. Good for CI testing.

## tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## project structure

```
conti/
  config_schema.py  — all experiment knobs (dataclass configs)
  reproducibility.py — seed locking, run manifests
  logging.py — JSONL + wandb logging
  data/
    loaders.py — math dataset + safety benchmark loaders
    advbench_subset.jsonl, xstest_subset.jsonl, donotanswer_subset.jsonl
  eval/
    metrics.py — pass@1, ASR proxy, multi-benchmark eval
    stats.py — aggregate mean/std/CI across seeds
    drift.py — per-round safety drift tracker
  loop/
    run.py — main self-improvement loop (the heart of everything)
  replay/
    buffer.py — safety replay buffer
  training/
    sft.py — supervised fine-tuning via Accelerate
    format.py — chat template formatting
    ewc.py — Elastic Weight Consolidation
  verifier/
    keyword.py — regex/keyword safety gate + composite verifier
    model_scorer.py — classifier-based safety scorer
configs/
  default.yaml — full config for Llama runs
  smoke_gpt2.yaml — tiny config for testing
scripts/
  run_experiment.py — single experiment entry point
  run_sweep.py — multi-seed sweep runner
  aggregate_results.py — collect metrics into summary table
  plot_results.py — generate publication figures
```

## references

- Zelikman et al., 2022. STaR: Bootstrapping Reasoning With Reasoning
- Qi et al., 2023. Fine-tuning Aligned LLMs Compromises Safety
- Askell et al., 2021. A General Language Assistant as a Laboratory for Alignment
- Zweiger et al., 2025. Self-adapting Language Models

Protocol details for collaborators in `docs/RESEARCH_PROTOCOL.md`.
