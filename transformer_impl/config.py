import yaml
import copy
from dataclasses import dataclass, field, asdict
from typing import Literal, Any


@dataclass
class AttentionConfig:
    type: Literal["mha", "mqa", "gqa", "linear", "window", "dilated", "global_local", "mamba", "ssm"] = "mha"
    num_heads: int = 8
    num_kv_heads: int | None = None
    window_size: int | None = None
    dilation: int | None = None
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4
    scale_attention: bool = True

@dataclass
class FFNConfig:
    type: Literal["standard", "swiglu", "gated", "moe"] = "standard"
    d_ff: int = 1024
    activation: str = "gelu"
    num_experts: int = 8
    top_k: int = 2

@dataclass
class PositionConfig:
    type: Literal["sinusoidal", "rope", "none"] = "sinusoidal"
    max_len: int = 5000
    rope_theta: float = 10000.0

@dataclass
class LossConfig:
    type: Literal["cross_entropy", "nll", "mse", "focal"] = "cross_entropy"
    label_smoothing: float = 0.025
    ignore_index: str = "auto"
    moe_load_balance_coef: float = 0.01
    focal_gamma: float | None = None

@dataclass
class TrainingStageMixin:
    gradient_accumulation_steps: int = 1
    mixed_precision: str | None = "bf16"
    logging: str = "tensorboard"
    log_interval: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    warmup_steps: int = 0
    max_steps: int = 100000
    save_total_limit: int = 3

@dataclass
class PretrainConfig(TrainingStageMixin):
    enabled: bool = False
    warmup_steps: int = 2000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    mixed_precision: str | None = "bf16"
    logging: str = "tensorboard"
    save_steps: int = 1000
    eval_steps: int = 1000
    log_interval: int = 10
    save_total_limit: int = 3
    resume_from: str | None = None
    streaming: bool = False

@dataclass
class SFTConfig(TrainingStageMixin):
    enabled: bool = False
    loss_on_response_only: bool = True
    packing: bool = True
    gradient_accumulation_steps: int = 1
    mixed_precision: str | None = None
    logging: str = "tensorboard"
    log_interval: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    warmup_steps: int = 0
    max_steps: int = 100000

@dataclass
class DPOConfig(TrainingStageMixin):
    enabled: bool = False
    beta: float = 0.1
    gradient_accumulation_steps: int = 1
    mixed_precision: str | None = None
    logging: str = "tensorboard"
    log_interval: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    warmup_steps: int = 0
    max_steps: int = 100000

@dataclass
class PPOConfig(TrainingStageMixin):
    enabled: bool = False
    kl_coef: float = 0.02
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_gen_length: int = 256
    gradient_accumulation_steps: int = 1
    mixed_precision: str | None = None
    logging: str = "tensorboard"
    log_interval: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    warmup_steps: int = 0
    max_steps: int = 100000

@dataclass
class GRPOConfig(TrainingStageMixin):
    enabled: bool = False
    group_size: int = 8
    epsilon: float = 0.2
    kl_coef: float = 0.01
    max_gen_length: int = 1024
    reward_fn: str = "exact_match"
    reward_fn_config: dict | None = None
    gradient_accumulation_steps: int = 1
    mixed_precision: str | None = None
    logging: str = "tensorboard"
    log_interval: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    warmup_steps: int = 0
    max_steps: int = 100000

@dataclass
class ModelConfig:
    d_model: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    attention_dropout: float | None = None
    ffn_dropout: float | None = None
    embedding_dropout: float | None = None
    stochastic_depth: float = 0.0
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    position: PositionConfig = field(default_factory=PositionConfig)

@dataclass
class DatasetConfig:
    name: str = "tinyshakespeare"
    path: str | None = None
    tokenization: str = "bpe"
    vocab_size: int = 4096
    max_seq_len: int = 128
    train_stride: int = 16
    cache_dir: str = "./data"
    max_train_chunks: int = 35000
    max_test_chunks: int = 2000

@dataclass
class TrainConfig:
    batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    early_stop_patience: int = 0
    warmup_steps: int | None = None
    max_steps: int | None = None
    gradient_accumulation_steps: int | None = None
    mixed_precision: str | None = None
    save_steps: int | None = None
    eval_steps: int | None = None
    logging: str | None = None
    resume_from: str | None = None
    loss: LossConfig = field(default_factory=LossConfig)

@dataclass
class ExperimentConfig:
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    seed: int = 42


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce(value, target_type):
    if target_type is float and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if target_type is int and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    if target_type in (float, int) and isinstance(value, bool):
        return int(value) if target_type is int else float(value)
    return value


def dict_to_dataclass(dc_type, data: dict):
    field_types = {}
    for f in dc_type.__dataclass_fields__.values():
        field_types[f.name] = f.type

    kwargs = {}
    for key, value in data.items():
        if key in field_types:
            ft = field_types[key]
            if hasattr(ft, '__dataclass_fields__') and isinstance(value, dict):
                kwargs[key] = dict_to_dataclass(ft, value)
            elif isinstance(value, dict):
                for f in ft.__args__ if hasattr(ft, '__args__') else []:
                    if hasattr(f, '__dataclass_fields__'):
                        kwargs[key] = dict_to_dataclass(f, value)
                        break
                else:
                    kwargs[key] = value
            else:
                kwargs[key] = _coerce(value, ft)
    return dc_type(**kwargs)


def config_from_dict(data: dict) -> ExperimentConfig:
    return dict_to_dataclass(ExperimentConfig, data)


def config_from_yaml(path: str) -> ExperimentConfig:
    data = load_yaml(path)
    return config_from_dict(data)


def config_from_cli(yaml_path: str | None = None, overrides: dict | None = None) -> ExperimentConfig:
    base = {}
    if yaml_path:
        base = load_yaml(yaml_path)
    if overrides:
        base = deep_merge(base, overrides)
    return config_from_dict(base)


def parse_cli_overrides(args: list[str]) -> dict:
    result = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        try:
            parsed = yaml.safe_load(value)
            d[parts[-1]] = parsed
        except Exception:
            d[parts[-1]] = value
    return result


def make_run_name(cfg, prefix="", timestamp=None):
    from datetime import datetime
    arch = f"{cfg.model.attention.type}_{cfg.model.ffn.type}_{cfg.model.position.type}"
    dims = f"d{cfg.model.d_model}l{cfg.model.num_layers}"
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [p for p in [prefix, arch, dims, ts] if p]
    return "_".join(parts)


def flatten_config(cfg: ExperimentConfig, prefix="") -> dict[str, Any]:
    flat = {}
    for key, value in asdict(cfg).items():
        if isinstance(value, dict):
            flat.update(flatten_dict(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def flatten_dict(d: dict, prefix="") -> dict[str, Any]:
    flat = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat.update(flatten_dict(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat
