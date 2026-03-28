from conti.eval.drift import DriftTracker


def test_basic():
    t = DriftTracker()
    t.set_baseline({"advbench": 0.05, "xstest": 0.02})
    pts = t.record(0, {"advbench": 0.15, "xstest": 0.08})
    assert len(pts) == 2

    adv = [p for p in pts if p.benchmark == "advbench"][0]
    assert abs(adv.absolute_drift - 0.10) < 1e-6


def test_no_baseline():
    t = DriftTracker()
    pts = t.record(0, {"advbench": 0.10})
    assert pts[0].baseline_asr == 0.0
    assert pts[0].absolute_drift == 0.10


def test_worst():
    t = DriftTracker()
    t.set_baseline({"advbench": 0.05, "xstest": 0.02})
    t.record(0, {"advbench": 0.40, "xstest": 0.10})
    w = t.get_worst_drift()
    assert w.benchmark == "advbench"


def test_summary():
    t = DriftTracker()
    t.set_baseline({"advbench": 0.05})
    t.record(0, {"advbench": 0.10})
    t.record(1, {"advbench": 0.20})
    s = t.to_summary_dict()
    assert s["total_points"] == 2
    assert s["last_round"] == 1


def test_empty():
    t = DriftTracker()
    assert t.get_worst_drift() is None
    assert t.to_summary_dict()["total_points"] == 0


def test_round_summary():
    t = DriftTracker()
    t.set_baseline({"advbench": 0.05, "xstest": 0.02})
    t.record(0, {"advbench": 0.12, "xstest": 0.05})
    rs = t.get_round_summary(0)
    assert "advbench" in rs
    assert "xstest" in rs
