from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class SafetyReplayBuffer:
    def __init__(self, path: Any | None, seed: int):
        self._items: list[dict[str, Any]] = []
        self._rng = random.Random(seed)
        if path is None:
            return
        text = self._read_all(path)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            self._items.append(json.loads(line))

    @staticmethod
    def _read_all(path: Any) -> str:
        # importlib.resources objects have read_text, pathlib has read_text,
        # god knows what else gets passed in here so try everything
        read_text = getattr(path, "read_text", None)
        if callable(read_text):
            return str(read_text(encoding="utf-8"))
        open_fn = getattr(path, "open", None)
        if callable(open_fn):
            with open_fn("r", encoding="utf-8") as fh:
                return fh.read()
        return Path(path).read_text(encoding="utf-8")

    def __len__(self) -> int:
        return len(self._items)

    # randomly uthao kuch items buffer se, training mein mix karenge
    def sample(self, n: int) -> list[dict[str, Any]]:
        if not self._items or n <= 0:
            return []
        n = min(n, len(self._items))
        return [self._rng.choice(self._items) for _ in range(n)]

    def add(self, item: dict[str, Any]) -> None:
        self._items.append(item)

    @property
    def size(self) -> int:
        return len(self._items)
