from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class RunManifest:
    run_id: str
    started_at_utc: str
    python: str
    argv: list[str]
    env_cuda_visible: str | None
    seed: int
    config_sha256: str
    config_dict: dict[str, Any]
    package_versions: dict[str, str]
    git_commit: str | None = None
    git_dirty: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_git_meta(repo_root: Path | None = None) -> tuple[str | None, bool | None]:
    root = repo_root or Path.cwd()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            != ""
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, None


def collect_package_versions() -> dict[str, str]:
    names = ["torch", "transformers", "accelerate", "datasets", "peft", "numpy"]
    out: dict[str, str] = {}
    for name in names:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            out[name] = str(ver)
        except ImportError:
            out[name] = "not_installed"
    out["python"] = sys.version.split()[0]
    return out


# sab jagah same seed set karo taki results reproduce ho
def set_global_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # full determinism tanks throughput, only flip on when auditing
    if os.environ.get("CONTI_DETERMINISTIC", "").lower() in ("1", "true", "yes"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_manifest(
    config_dict: dict[str, Any],
    seed: int,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RunManifest:
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit, dirty = get_git_meta()
    return RunManifest(
        run_id=rid,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.executable,
        argv=list(sys.argv),
        env_cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        seed=seed,
        config_sha256=sha256_json(config_dict),
        config_dict=config_dict,
        package_versions=collect_package_versions(),
        git_commit=commit,
        git_dirty=dirty,
        extra=extra or {},
    )
