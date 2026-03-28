# Research Protocol

This protocol is designed so results are credible in a lab discussion and reproducible by collaborators.

## core hypotheses

- **H1:** Naive continual self-improvement degrades safety metrics over rounds (the model forgets its RLHF guardrails as it trains on self-generated math trajectories)
- **H2:** Batch-level verifier filtering reduces this safety drift
- **H3:** Verifier + replay buffer + EWC achieves better safety retention with smaller capability tax than strict filtering alone

## experiment arms (locked to proposal)

| Arm | What it tests |
|-----|--------------|
| `baseline_frozen` | Seed model unchanged — establishes ground truth scores |
| `baseline_single_sft` | One-shot SFT on static math data — shows static weight drift cost |
| `naive_continual` | Recursive self-improvement, no safety filters — shows compounding drift |
| `phase1_verifier` | Self-improvement + verifier filtering — isolates verifier contribution |
| `phase2_verifier_buffer` | Verifier + safety replay buffer — full treatment |

### protocol guards
Code-level guards raise errors for invalid configs (e.g. replay outside phase2, single_sft with >1 round).

## baseline semantics

- `baseline_frozen`: no training, just eval
- `baseline_single_sft`: exactly 1 round, static human-verified trajectories only
- `naive_continual`: recursive STaR-style loop, correctness-gated but NO safety filtering
- `phase1_verifier`: same loop + keyword/model verifier filtering before SFT
- `phase2_verifier_buffer`: phase1 + safety replay buffer mixed into training batches

Self-improvement trajectories are always correctness-gated (`require_correct_trajectory=true`) for consistency with STaR.

## statistical design

- Run each arm with **≥3 seeds** (prefer 5 for paper)
- Same base config except `seed` and `loop.experiment`
- Aggregate with mean/std and 95% CI across seeds
- Wilson intervals for proportion metrics (better for small n)

## primary metrics

### capability
- Math Pass@1 accuracy on GSM8K/MATH held-out test set

### safety
- **AdvBench ASR** — how often the model complies with harmful prompts
- **XSTEST** — over-refusal and under-refusal on ambiguous prompts
- **do-not-answer** — broad safety coverage
- **Aggregate ASR** — weighted average across benchmarks

### training dynamics
- `accepted_train_examples` — how many trajectories passed filters
- `rejected_by_verifier` — how many were caught by safety gate
- `rejected_by_correctness` — how many were wrong (STaR gate)
- Per-round safety drift from baseline (absolute + relative)

## audit artifacts

Each run must include:
- `run_manifest.json` — config hash, versions, environment, git state
- `metrics.json` — final math + safety metrics + drift summary
- `logs/round_metrics.jsonl` — per-round evaluations
- `logs/filter_log.jsonl` — every verifier decision
- `logs/drift_log.jsonl` — per-round safety drift measurements
- `logs/training_steps.jsonl` — per-step loss and EWC penalty
- `checkpoints/round_N/` — model weights after each round

## suggested presentation table

| Arm | Math Pass@1 (mean ± 95% CI) | AdvBench ASR | XSTEST ASR | do-not-answer ASR | Aggregate ASR |
|-----|---|---|---|---|---|
| baseline_frozen | TODO | TODO | TODO | TODO | TODO |
| baseline_single_sft | TODO | TODO | TODO | TODO | TODO |
| naive_continual | TODO | TODO | TODO | TODO | TODO |
| phase1_verifier | TODO | TODO | TODO | TODO | TODO |
| phase2_verifier_buffer | TODO | TODO | TODO | TODO | TODO |

### recommended figures
1. Safety drift curves: ASR over rounds for naive vs verifier vs verifier+buffer
2. Capability-safety Pareto: math accuracy vs aggregate ASR per arm
3. Training dynamics: accepted/rejected examples per round

## integrity policy

- Never hand-edit metric files
- Never report unrun experiments
- Keep failed seeds in appendices and explain exclusions explicitly
- Run manifests provide cryptographic audit trail (config hash)
