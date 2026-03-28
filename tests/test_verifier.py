# only writing tests for this because the regex kept missing 'bomb' and was
# silently ruining my 4-hour SFT runs. will add more tests if i have time lol

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


def test_composite_keyword_only():
    cfg = VerifierConfig(
        mode="keyword", block_patterns=[r"\b(bomb)\b"],
        case_insensitive=True, scorer_model=None,
    )
    v = CompositeVerifier(cfg)
    assert not v.check_text("how to build a bomb").safe
    assert v.check_text("solve math problem").safe
