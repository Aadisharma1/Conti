import tempfile
from pathlib import Path

from conti.config_schema import ContiConfig, load_config, dict_to_config


def test_defaults():
    cfg = ContiConfig()
    assert cfg.seed == 42
    assert cfg.model.name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert cfg.verifier.mode == "keyword"
    assert cfg.ewc.enabled is False
    assert cfg.logging.use_wandb is False


def test_empty_dict():
    cfg = dict_to_config({})
    assert cfg.seed == 42
    assert cfg.output_dir == "./outputs/run"


def test_overrides():
    cfg = dict_to_config({
        "seed": 123,
        "model": {"name_or_path": "gpt2", "use_lora": False},
        "ewc": {"enabled": True, "lambda_ewc": 1000.0},
    })
    assert cfg.seed == 123
    assert cfg.model.name_or_path == "gpt2"
    assert not cfg.model.use_lora
    assert cfg.ewc.enabled
    assert cfg.ewc.lambda_ewc == 1000.0


def test_yaml_load():
    yaml_str = """
seed: 77
output_dir: ./outputs/test
model:
  name_or_path: gpt2
  torch_dtype: float32
loop:
  experiment: baseline_frozen
  num_self_improve_rounds: 1
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_str)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.seed == 77
    assert cfg.model.name_or_path == "gpt2"
    assert cfg.loop.experiment == "baseline_frozen"


def test_roundtrip():
    d = ContiConfig().to_dict()
    assert isinstance(d, dict)
    assert d["seed"] == 42


def test_three_benchmarks():
    cfg = ContiConfig()
    assert "advbench_subset" in cfg.data.safety_eval_datasets
    assert "xstest_subset" in cfg.data.safety_eval_datasets
    assert "donotanswer_subset" in cfg.data.safety_eval_datasets
