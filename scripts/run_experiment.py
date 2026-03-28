#!/usr/bin/env python3
# run_experiment.py — main entry point for experiments
# works with both single GPU and `accelerate launch` for multi-GPU

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="conti safety — continual learning experiments")
    parser.add_argument("--config", type=str, required=True, help="path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="override output dir from config")
    parser.add_argument("--seed", type=int, default=None, help="override seed from config")
    args = parser.parse_args()

    from conti.config_schema import load_config
    from conti.loop.run import run_experiment

    cfg = load_config(args.config)

    # env var override for containers / cloud
    if root := os.environ.get("CONTI_OUTPUT_DIR"):
        cfg.output_dir = root
    # CLI overrides
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed

    run_experiment(cfg)


if __name__ == "__main__":
    main()
