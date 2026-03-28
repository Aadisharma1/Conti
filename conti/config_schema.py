from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

T = TypeVar("T")


def _merge_dataclass(cls: type[T], overrides: dict[str, Any]) -> T:
    base = asdict(cls())
    names = {f.name for f in fields(cls)}
    merged = {k: v for k, v in {**base, **overrides}.items() if k in names}
    return cls(**merged)


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    attn_implementation: str | None = None  # flash attention lagao agar GPU support kare toh


@dataclass
class VerifierConfig:
    mode: Literal["keyword", "keyword_and_model"] = "keyword"
    block_patterns: list[str] = field(
        default_factory=lambda: [
            r"\b(bomb|weapon|kill|suicide)\b",
            r"\b(how to (make|build) (a )?(meth|explosive))\b",
        ]
    )
    case_insensitive: bool = True
    scorer_model: str | None = None
    scorer_threshold: float = 0.75


@dataclass
class ReplayBufferConfig:
    enabled: bool = False
    safety_samples_per_batch: int = 4
    dataset_path: str | None = None


@dataclass
class EWCConfig:
    # fisher info se important weights ko lock karo basically
    enabled: bool = False
    lambda_ewc: float = 5000.0  # treat as hyperparam
    fisher_samples: int = 200


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "conti-safety"
    wandb_entity: str | None = None
    log_every_n_steps: int = 10


@dataclass
class LoopConfig:
    experiment: Literal[
        "baseline_frozen",
        "baseline_single_sft",
        "naive_continual",
        "phase1_verifier",
        "phase2_verifier_buffer",
    ] = "phase1_verifier"
    num_self_improve_rounds: int = 2
    generate_batch_size: int = 4
    train_micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_epochs_per_round: int = 1
    max_train_samples_per_round: int | None = 500  # zyada samples = zyada time, adjust karna padega
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    generation_temperature: float = 0.7
    generation_top_p: float = 0.95
    generation_max_new_tokens: int = 512
    eval_every_round: bool = True
    require_correct_trajectory: bool = True
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    math_dataset: str = "gsm8k"
    math_split: str = "train"
    math_test_split: str = "test"
    safety_eval_datasets: list[str] = field(
        default_factory=lambda: ["advbench_subset", "xstest_subset", "donotanswer_subset"]
    )
    max_prompts_per_eval: int = 100


@dataclass
class ContiConfig:
    seed: int = 42
    output_dir: str = "./outputs/run"
    model: ModelConfig = field(default_factory=ModelConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    ewc: EWCConfig = field(default_factory=EWCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ContiConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return dict_to_config(raw)


def dict_to_config(d: dict[str, Any]) -> ContiConfig:
    m = d.get("model") or {}
    v = d.get("verifier") or {}
    r = d.get("replay") or {}
    e = d.get("ewc") or {}
    lg = d.get("logging") or {}
    lp = d.get("loop") or {}
    dt = d.get("data") or {}
    return ContiConfig(
        seed=int(d.get("seed", 42)),
        output_dir=str(d.get("output_dir", "./outputs/run")),
        model=_merge_dataclass(ModelConfig, m),
        verifier=_merge_dataclass(VerifierConfig, v),
        replay=_merge_dataclass(ReplayBufferConfig, r),
        ewc=_merge_dataclass(EWCConfig, e),
        logging=_merge_dataclass(LoggingConfig, lg),
        loop=_merge_dataclass(LoopConfig, lp),
        data=_merge_dataclass(DataConfig, dt),
    )
