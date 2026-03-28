from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


@dataclass
class MetricSummary:
    name: str
    values: list[float]
    n: int
    mean: float
    std: float
    ci95_half_width: float

    def to_dict(self) -> dict[str, float | int | list[float] | str]:
        return {
            "name": self.name,
            "values": self.values,
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "ci95_half_width": self.ci95_half_width,
        }


def summarize(values: list[float], name: str) -> MetricSummary:
    n = len(values)
    if n == 0:
        return MetricSummary(name=name, values=[], n=0, mean=0.0, std=0.0, ci95_half_width=0.0)
    mu = mean(values)
    sd = stdev(values) if n > 1 else 0.0
    ci = 1.96 * (sd / (n**0.5)) if n > 1 else 0.0
    return MetricSummary(name=name, values=values, n=n, mean=mu, std=sd, ci95_half_width=ci)


def collect_metric_values(run_dirs: list[Path], metric_path: list[str]) -> list[float]:
    out: list[float] = []
    for d in run_dirs:
        p = d / "metrics.json"
        if not p.exists():
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        cur = obj
        try:
            for k in metric_path:
                cur = cur[k]
            out.append(float(cur))
        except (KeyError, TypeError, ValueError):
            continue
    return out
