#!/usr/bin/env python3
# aggregate_results.py — collect metrics from multiple runs into a summary
# used after run_sweep.py to get mean/std/CI across seeds

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conti.eval.stats import collect_metric_values, summarize


def _find_run_dirs(root: Path, prefix: str) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)])


def main() -> None:
    parser = argparse.ArgumentParser(description="aggregate conti experiment metrics")
    parser.add_argument("--root", required=True, help="root directory containing runs")
    parser.add_argument("--experiments", nargs="+", required=True, help="experiment name prefixes")
    parser.add_argument("--out", default=None, help="optional output JSON path")
    args = parser.parse_args()

    root = Path(args.root)
    report: dict[str, dict] = {}

    for exp in args.experiments:
        dirs = _find_run_dirs(root, f"{exp}_seed")
        math_vals = collect_metric_values(dirs, ["math_pass1", "accuracy"])
        # updated path for multi-benchmark: aggregate ASR
        asr_vals = collect_metric_values(dirs, ["safety_asr", "aggregate", "asr_proxy"])
        # also grab per-benchmark ASR if available
        advbench_vals = collect_metric_values(dirs, ["safety_asr", "advbench_subset", "asr_proxy"])

        report[exp] = {
            "num_runs_found": len(dirs),
            "math_pass1_accuracy": summarize(math_vals, "math_pass1_accuracy").to_dict(),
            "safety_asr_aggregate": summarize(asr_vals, "safety_asr_aggregate").to_dict(),
            "safety_asr_advbench": summarize(advbench_vals, "safety_asr_advbench").to_dict(),
        }

    payload = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"saved to {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
