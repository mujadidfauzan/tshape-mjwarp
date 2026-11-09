"""Validated MAPPO settings, independent of environment configuration."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import math
from numbers import Real
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "configs" / "mappo.yaml"


@dataclass(frozen=True)
class MAPPOConfig:
    seed: int = 42
    device: str = "cpu"
    total_steps: int = 100000
    rollout_steps: int = 1024
    ppo_epochs: int = 5
    num_minibatches: int = 1
    actor_hidden_size: int = 128
    critic_hidden_size: int = 256
    actor_lr: float = 0.0003
    critic_lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    initial_log_std: float = -0.5
    normalize_inputs: bool = True
    normalization_clip: float = 10.0
    checkpoint_interval_steps: int = 10000
    eval_interval_steps: int = 10000
    eval_episodes: int = 10
    torch_threads: int = 1
    video_interval_steps: int = 100000
    video_width: int = 1920
    video_height: int = 1080
    video_crf: int = 18
    tensorboard: bool = True
    stats_window_size: int = 100
    reward_log_interval_steps: int = 100

    def __post_init__(self):
        integer_fields = ("total_steps", "rollout_steps", "ppo_epochs", "num_minibatches",
                          "actor_hidden_size", "critic_hidden_size", "checkpoint_interval_steps",
                          "eval_interval_steps", "eval_episodes", "torch_threads", "stats_window_size")
        for name in (*integer_fields, "seed"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.num_minibatches > self.rollout_steps:
            raise ValueError("num_minibatches cannot exceed rollout_steps")
        for name in ("video_interval_steps", "video_width", "video_height", "video_crf", "reward_log_interval_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if any(value < 2 or value % 2 for value in (self.video_width, self.video_height)):
            raise ValueError("video dimensions must be even integers >= 2")
        if self.video_crf > 51:
            raise ValueError("video_crf must lie in [0, 51]")
        if not isinstance(self.normalize_inputs, bool):
            raise ValueError("normalize_inputs must be boolean")
        if not isinstance(self.tensorboard, bool):
            raise ValueError("tensorboard must be boolean")
        if not isinstance(self.device, str) or (self.device not in ("cpu", "auto", "cuda")
                                                and not (self.device.startswith("cuda:")
                                                         and self.device[5:].isdigit())):
            raise ValueError("device must be cpu, auto, cuda, or cuda:<index>")
        for name in ("actor_lr", "critic_lr", "gamma", "gae_lambda", "clip_ratio",
                     "value_clip", "value_coef", "entropy_coef", "max_grad_norm",
                     "target_kl", "initial_log_std", "normalization_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        for name in ("actor_lr", "critic_lr", "max_grad_norm", "normalization_clip", "target_kl"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must lie in (0, 1)")
        for name in ("value_clip", "value_coef", "entropy_coef"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not -5 <= self.initial_log_std <= 2:
            raise ValueError("initial_log_std must lie in [-5, 2]")

    def to_dict(self):
        return asdict(self)


def load_mappo_config(config=None, **overrides):
    if isinstance(config, MAPPOConfig):
        result = config
    else:
        if config is None:
            config = DEFAULT_PATH
        if isinstance(config, (str, Path)):
            with Path(config).open(encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
        if not isinstance(config, Mapping):
            raise ValueError("MAPPO configuration must be a mapping")
        config = dict(config)
        unknown = set(config) - {field.name for field in fields(MAPPOConfig)}
        if unknown:
            raise ValueError(f"unknown MAPPO keys: {sorted(unknown)}")
        result = MAPPOConfig(**config)
    return replace(result, **{name: value for name, value in overrides.items() if value is not None})
