from conti.eval.stats import summarize


def test_basic():
    s = summarize([0.2, 0.3, 0.5], "x")
    assert s.n == 3
    assert 0.0 < s.mean < 1.0
    assert s.std > 0.0
    assert s.ci95_half_width > 0.0


def test_single():
    s = summarize([0.42], "solo")
    assert s.n == 1
    assert s.mean == 0.42
    assert s.std == 0.0


def test_empty():
    s = summarize([], "empty")
    assert s.n == 0
    assert s.mean == 0.0


def test_to_dict():
    s = summarize([0.1, 0.2, 0.3], "test")
    d = s.to_dict()
    assert d["name"] == "test"
    assert d["n"] == 3


def test_identical():
    s = summarize([0.5, 0.5, 0.5], "same")
    assert s.mean == 0.5
    assert s.std == 0.0
