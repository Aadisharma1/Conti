from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from conti.config_schema import ContiConfig


class ExperimentLogger:
    def __init__(self, cfg: ContiConfig, run_dir: str | Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self._log_dir = self.run_dir / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._step_log = self._log_dir / "training_steps.jsonl"
        self._round_log = self._log_dir / "round_metrics.jsonl"

        self._wandb_run = None
        if cfg.logging.use_wandb and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=cfg.logging.wandb_project,
                    entity=cfg.logging.wandb_entity,
                    config=cfg.to_dict(),
                    name=f"{cfg.loop.experiment}_seed{cfg.seed}",
                    reinit=True,
                )
            except Exception as e:
                print(f"[warn] wandb init failed: {e}")

    def log_step(self, metrics: dict[str, Any], step: int) -> None:
        entry = {"step": step, **metrics}
        with self._step_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def log_round(self, round_idx: int, metrics: dict[str, Any]) -> None:
        entry = {"round": round_idx, **metrics}
        with self._round_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self._wandb_run is not None:
            try:
                import wandb
                flat = self._flatten(metrics, prefix="round")
                flat["round_idx"] = round_idx
                wandb.log(flat)
            except Exception:
                pass

    def log_drift(self, round_idx: int, drift_data: dict[str, Any]) -> None:
        entry = {"round": round_idx, "drift": drift_data}
        drift_log = self._log_dir / "drift_log.jsonl"
        with drift_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self._wandb_run is not None:
            try:
                import wandb
                flat = self._flatten(drift_data, prefix="drift")
                flat["round_idx"] = round_idx
                wandb.log(flat)
            except Exception:
                pass

    def finish(self) -> None:
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    @staticmethod
    def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                out.update(ExperimentLogger._flatten(v, key))
            else:
                out[key] = v
        return out
