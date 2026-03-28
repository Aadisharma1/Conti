from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class DriftPoint:
    round_idx: int
    benchmark: str
    baseline_asr: float
    current_asr: float
    absolute_drift: float
    relative_drift: float


@dataclass
class DriftTracker:
    baseline_scores: dict[str, float] = field(default_factory=dict)
    history: list[DriftPoint] = field(default_factory=list)

    def set_baseline(self, scores: dict[str, float]) -> None:
        self.baseline_scores = dict(scores)

    def record(self, round_idx: int, current_scores: dict[str, float]) -> list[DriftPoint]:
        points = []
        for bench, cur_asr in current_scores.items():
            base_asr = self.baseline_scores.get(bench, 0.0)
            abs_drift = cur_asr - base_asr
            # agar baseline 0 hai toh relative drift infinity hoga, cap it
            rel_drift = abs_drift / max(base_asr, 1e-8)
            pt = DriftPoint(
                round_idx=round_idx,
                benchmark=bench,
                baseline_asr=base_asr,
                current_asr=cur_asr,
                absolute_drift=abs_drift,
                relative_drift=rel_drift,
            )
            points.append(pt)
            self.history.append(pt)
        return points

    def get_worst_drift(self) -> DriftPoint | None:
        if not self.history:
            return None
        return max(self.history, key=lambda p: p.absolute_drift)

    def get_round_summary(self, round_idx: int) -> dict[str, float]:
        pts = [p for p in self.history if p.round_idx == round_idx]
        if not pts:
            return {}
        return {
            p.benchmark: {
                "abs_drift": p.absolute_drift,
                "rel_drift": p.relative_drift,
                "current_asr": p.current_asr,
            }
            for p in pts
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for pt in self.history:
                f.write(json.dumps(asdict(pt), ensure_ascii=False) + "\n")

    def to_summary_dict(self) -> dict[str, Any]:
        if not self.history:
            return {"total_points": 0}
        last_round = max(p.round_idx for p in self.history)
        last_pts = [p for p in self.history if p.round_idx == last_round]
        worst = self.get_worst_drift()
        return {
            "total_points": len(self.history),
            "last_round": last_round,
            "final_drift": {p.benchmark: p.absolute_drift for p in last_pts},
            "worst_benchmark": worst.benchmark if worst else None,
            "worst_abs_drift": worst.absolute_drift if worst else 0.0,
        }
