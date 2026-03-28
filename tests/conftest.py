import pytest
from conti.config_schema import ContiConfig, VerifierConfig


@pytest.fixture
def default_cfg():
    return ContiConfig()


@pytest.fixture
def verifier_cfg():
    return VerifierConfig(
        block_patterns=[r"\b(bomb|weapon)\b"],
        case_insensitive=True,
    )


@pytest.fixture
def smoke_cfg():
    cfg = ContiConfig()
    cfg.model.name_or_path = "gpt2"
    cfg.model.torch_dtype = "float32"
    cfg.model.use_lora = False
    cfg.model.max_seq_length = 256
    cfg.loop.max_train_samples_per_round = 4
    cfg.loop.generate_batch_size = 2
    cfg.loop.num_self_improve_rounds = 1
    return cfg
