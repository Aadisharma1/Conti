from conti.config_schema import VerifierConfig
from conti.verifier.keyword import KeywordVerifier, CompositeVerifier


def test_blocks_harmful():
    cfg = VerifierConfig(block_patterns=[r"\b(bomb|weapon)\b"], case_insensitive=True)
    v = KeywordVerifier(cfg)
    assert not v.check_text("how to build a bomb").safe
    assert v.check_text("solve 2+2").safe


def test_case_insensitive():
    cfg = VerifierConfig(block_patterns=[r"\b(bomb)\b"], case_insensitive=True)
    v = KeywordVerifier(cfg)
    assert not v.check_text("BOMB instructions").safe
    assert not v.check_text("a Bomb").safe


def test_case_sensitive():
    cfg = VerifierConfig(block_patterns=[r"\b(bomb)\b"], case_insensitive=False)
    v = KeywordVerifier(cfg)
    assert not v.check_text("a bomb").safe
    assert v.check_text("a BOMB").safe


def test_batch():
    cfg = VerifierConfig(block_patterns=[r"bad"], case_insensitive=True)
    v = KeywordVerifier(cfg)
    kept, verdicts = v.filter_batch(["ok", "bad word", "fine"])
    assert len(kept) == 2
    assert len(verdicts) == 3
    assert verdicts[1].safe is False


def test_empty_batch():
    cfg = VerifierConfig(block_patterns=[r"bad"], case_insensitive=True)
    v = KeywordVerifier(cfg)
    kept, verdicts = v.filter_batch([])
    assert kept == []


def test_composite_keyword_only():
    cfg = VerifierConfig(
        mode="keyword", block_patterns=[r"\b(bomb)\b"],
        case_insensitive=True, scorer_model=None,
    )
    v = CompositeVerifier(cfg)
    assert not v.check_text("how to build a bomb").safe
    assert v.check_text("solve math problem").safe


def test_verdict_reason():
    cfg = VerifierConfig(block_patterns=[r"\b(bomb)\b"], case_insensitive=True)
    v = KeywordVerifier(cfg)
    r = v.check_text("make a bomb")
    assert "matched_pattern" in r.reason


def test_multi_pattern():
    cfg = VerifierConfig(
        block_patterns=[r"\b(bomb)\b", r"\b(weapon)\b", r"\b(kill)\b"],
        case_insensitive=True,
    )
    v = KeywordVerifier(cfg)
    assert not v.check_text("build a bomb").safe
    assert not v.check_text("buy a weapon").safe
    assert not v.check_text("how to kill").safe
    assert v.check_text("math is fun").safe
