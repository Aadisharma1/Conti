from conti.data.loaders import load_safety_prompts, load_all_safety_prompts, _parse_jsonl_prompts


def test_advbench():
    p = load_safety_prompts("advbench_subset", max_samples=100)
    assert len(p) > 0
    assert all(isinstance(x, str) for x in p)


def test_xstest():
    p = load_safety_prompts("xstest_subset", max_samples=100)
    assert len(p) > 0


def test_donotanswer():
    p = load_safety_prompts("donotanswer_subset", max_samples=100)
    assert len(p) > 0


def test_max_samples():
    full = load_safety_prompts("advbench_subset", max_samples=1000)
    limited = load_safety_prompts("advbench_subset", max_samples=3)
    assert len(limited) == 3
    assert len(limited) < len(full)


def test_multi():
    r = load_all_safety_prompts(
        ["advbench_subset", "xstest_subset", "donotanswer_subset"],
        max_per_benchmark=5,
    )
    assert len(r) == 3
    assert all(len(v) <= 5 for v in r.values())


def test_missing_benchmark():
    r = load_all_safety_prompts(["doesnt_exist_12345"], max_per_benchmark=5)
    assert r["doesnt_exist_12345"] == []


def test_parse_jsonl():
    raw = '{"prompt": "hello"}\n{"prompt": "world"}\n'
    p = _parse_jsonl_prompts(raw, key="prompt")
    assert p == ["hello", "world"]


def test_parse_max():
    raw = '{"prompt": "a"}\n{"prompt": "b"}\n{"prompt": "c"}\n'
    p = _parse_jsonl_prompts(raw, key="prompt", max_n=2)
    assert len(p) == 2
