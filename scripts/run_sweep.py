#!/usr/bin/env python3
# run_sweep.py — run all experiment arms across multiple seeds
#
# this generates one config per (arm, seed) combination and runs them
# sequentially. for massive sweeps you'd want to parallelize this
# with slurm or something, but for our 5 arms x 3-5 seeds its fine.

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _write_cfg(base_cfg: dict, out_path: Path, experiment: str, seed: int, output_dir: str) -> None:
    """generate a run-specific config from the base + overrides"""
    cfg = dict(base_cfg)
    cfg["seed"] = seed
    cfg["output_dir"] = output_dir
    cfg["loop"] = dict(base_cfg.get("loop", {}))
    cfg["loop"]["experiment"] = experiment

    # phase2 needs replay enabled
    if experiment == "phase2_verifier_buffer":
        cfg["replay"] = dict(base_cfg.get("replay", {}))
        cfg["replay"]["enabled"] = True
    else:
        cfg["replay"] = dict(base_cfg.get("replay", {}))
        cfg["replay"]["enabled"] = False

    # baseline_single_sft should be 1 round
    if experiment == "baseline_single_sft":
        cfg["loop"]["num_self_improve_rounds"] = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="conti seed sweep runner")
    parser.add_argument("--base-config", required=True, help="base YAML config")
    parser.add_argument("--out-root", required=True, help="output root for all runs")
    parser.add_argument("--experiments", nargs="+", required=True, help="experiment arms to run")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="random seeds")
    parser.add_argument("--accelerate", action="store_true", help="use accelerate launch")
    parser.add_argument("--dry-run", action="store_true", help="just generate configs, dont actually run")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    out_root = Path(args.out_root)
    cfg_root = out_root / "generated_configs"
    cfg_root.mkdir(parents=True, exist_ok=True)

    total_runs = len(args.experiments) * len(args.seeds)
    run_count = 0

    for exp in args.experiments:
        for seed in args.seeds:
            run_count += 1
            run_name = f"{exp}_seed{seed}"
            run_dir = out_root / run_name
            cfg_path = cfg_root / f"{run_name}.yaml"

            print(f"\n[{run_count}/{total_runs}] {run_name}")
            _write_cfg(base_cfg, cfg_path, exp, seed, str(run_dir))

            if args.dry_run:
                print(f"  config written to {cfg_path} (dry run, not executing)")
                continue

            cmd = [sys.executable, "scripts/run_experiment.py", "--config", str(cfg_path)]
            if args.accelerate:
                cmd = ["accelerate", "launch", "--multi_gpu"] + cmd

            print(f"  running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

    print(f"\ndone — {run_count} runs completed under {out_root}")


if __name__ == "__main__":
    main()
