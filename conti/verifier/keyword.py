from __future__ import annotations

import re
from dataclasses import dataclass

from conti.config_schema import VerifierConfig


@dataclass(frozen=True)
class Verdict:
    safe: bool
    reason: str


class KeywordVerifier:
    def __init__(self, cfg: VerifierConfig):
        flags = re.IGNORECASE if cfg.case_insensitive else 0
        self._patterns = [re.compile(p, flags) for p in cfg.block_patterns]

    # pehle regex check, match hua toh seedha reject
    def check_text(self, text: str) -> Verdict:
        for pat in self._patterns:
            if pat.search(text):
                return Verdict(safe=False, reason=f"matched_pattern:{pat.pattern[:80]}")
        return Verdict(safe=True, reason="ok")

    def filter_batch(self, texts: list[str]) -> tuple[list[str], list[Verdict]]:
        kept: list[str] = []
        verdicts: list[Verdict] = []
        for t in texts:
            v = self.check_text(t)
            verdicts.append(v)
            if v.safe:
                kept.append(t)
        return kept, verdicts


class CompositeVerifier:
    # NOTE: we running on hopes and vibes rn 
    # TODO: the scorer_threshold=0.5 is just vibes, should tune this on a held-out set

    def __init__(self, cfg: VerifierConfig):
        self._keyword = KeywordVerifier(cfg)
        self._model_scorer = None

        if cfg.mode == "keyword_and_model" and cfg.scorer_model:
            try:
                from conti.verifier.model_scorer import ModelScorer
                self._model_scorer = ModelScorer(
                    model_name=cfg.scorer_model,
                    threshold=cfg.scorer_threshold,
                )
            except Exception as e:
                print(f"[warn] model scorer failed, keyword only: {e}")

    def check_text(self, text: str) -> Verdict:
        v = self._keyword.check_text(text)
        if not v.safe:
            return v
        if self._model_scorer is not None:
            return self._model_scorer.check_text(text)
        return v

    def filter_batch(self, texts: list[str]) -> tuple[list[str], list[Verdict]]:
        kept: list[str] = []
        verdicts: list[Verdict] = []
        for t in texts:
            v = self.check_text(t)
            verdicts.append(v)
            if v.safe:
                kept.append(t)
        return kept, verdicts
