#!/usr/bin/env python3
# plot_results.py — generate publication-quality figures from experiment results
#
# this script reads the drift logs, round metrics, and aggregate results
# to produce the plots we need for the paper:
#   1. safety drift curves across experiment arms
#   2. capability (math acc) vs safety (ASR) pareto plot
#   3. per-round bar charts for accepted/rejected samples
#
# requires matplotlib. run after aggregate_results.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # no GUI needed
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# color palette — not the default matplotlib colors
# bc those look awful in papers
COLORS = {
    "baseline_frozen": "#2E86AB",
    "baseline_single_sft": "#A23B72",
    "naive_continual": "#F18F01",
    "phase1_verifier": "#C73E1D",
    "phase2_verifier_buffer": "#3B1F2B",
}

ARM_LABELS = {
    "baseline_frozen": "Frozen Seed",
    "baseline_single_sft": "Single-Shot SFT",
    "naive_continual": "Naive Continual",
    "phase1_verifier": "Phase 1 (Verifier)",
    "phase2_verifier_buffer": "Phase 2 (Verifier + Buffer)",
}


def plot_drift_curves(sweep_root: Path, experiments: list[str], out_path: Path) -> None:
    """plot safety drift over rounds for each experiment arm.
    this is THE key figure — shows naive degrading vs verifier stabilizing."""

    if not HAS_MPL:
        print("[warn] matplotlib not installed, skipping plots")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for exp in experiments:
        # find all runs for this arm
        run_dirs = sorted([p for p in sweep_root.iterdir()
                          if p.is_dir() and p.name.startswith(f"{exp}_seed")])
        if not run_dirs:
            continue

        # collect per-round aggregate ASR across seeds
        all_rounds: dict[int, list[float]] = {}
        for rd in run_dirs:
            drift_log = rd / "logs" / "drift_log.jsonl"
            if not drift_log.exists():
                continue
            for line in drift_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                pt = json.loads(line)
                if pt.get("benchmark") == "aggregate":
                    ridx = pt["round_idx"]
                    all_rounds.setdefault(ridx, []).append(pt["absolute_drift"])

        if not all_rounds:
            continue

        rounds = sorted(all_rounds.keys())
        means = [sum(all_rounds[r]) / len(all_rounds[r]) for r in rounds]
        color = COLORS.get(exp, "#333333")
        label = ARM_LABELS.get(exp, exp)
        ax.plot(rounds, means, "o-", color=color, label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Self-Improvement Round", fontsize=12)
    ax.set_ylabel("Safety Drift (ΔASR from Baseline)", fontsize=12)
    ax.set_title("Safety Drift Across Experiment Arms", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"saved drift plot to {out_path}")
    plt.close(fig)


def plot_capability_safety(aggregate_path: Path, out_path: Path) -> None:
    """pareto plot: math accuracy vs safety ASR for each arm.
    ideally we want top-right (high capability, low ASR)."""

    if not HAS_MPL:
        return

    data = json.loads(aggregate_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 6))

    for exp, metrics in data.items():
        math_acc = metrics.get("math_pass1_accuracy", {}).get("mean", 0)
        safety_asr = metrics.get("safety_asr_aggregate", {}).get("mean", 0)
        color = COLORS.get(exp, "#333333")
        label = ARM_LABELS.get(exp, exp)
        ax.scatter(math_acc, safety_asr, c=color, s=120, label=label, zorder=5, edgecolors="white")

    ax.set_xlabel("Math Pass@1 Accuracy", fontsize=12)
    ax.set_ylabel("Safety ASR (lower = better)", fontsize=12)
    ax.set_title("Capability vs Safety Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"saved pareto plot to {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="plot conti experiment results")
    parser.add_argument("--sweep-root", required=True, help="sweep output root")
    parser.add_argument("--aggregate", default=None, help="path to aggregate.json")
    parser.add_argument("--experiments", nargs="+", default=[
        "baseline_frozen", "baseline_single_sft", "naive_continual",
        "phase1_verifier", "phase2_verifier_buffer",
    ])
    parser.add_argument("--out-dir", default=None, help="output dir for plots")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = Path(args.out_dir) if args.out_dir else sweep_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_drift_curves(sweep_root, args.experiments, out_dir / "safety_drift.png")

    if args.aggregate:
        plot_capability_safety(Path(args.aggregate), out_dir / "capability_safety.png")

    print(f"\nall plots saved to {out_dir}")


if __name__ == "__main__":
    main()
