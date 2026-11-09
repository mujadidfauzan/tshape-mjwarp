"""Easy/Medium task settings with configurable agent counts and seeded spawns."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from numbers import Real
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs/environment.yaml"


@dataclass(frozen=True)
class EnvironmentConfig:
    model_path: str = "assets/mujoco/arena.xml"
    frame_skip: int = 50
    max_steps: int = 500
    num_agents: int = 4
    randomize_agents: bool = False  # Old snapshots omit this key; new runs enable it in YAML.
    spawn_gap_min: float = .04
    spawn_gap_max: float = .12
    object_pose: tuple = (-1.5, -1.5, 0.0)
    agent_positions: tuple | None = None
    goal_center: tuple = (1.5, 1.5)
    goal_half_size: tuple = (.75, .65)
    goal_yaw: float = 0.0
    success_linear_speed: float = .05
    success_angular_speed: float = .1
    success_hold_steps: int = 5
    distance_weight: float = .1
    orientation_weight: float = .05
    progress_weight: float = 2.0
    success_reward: float = 20.0
    distance_scale: float = 1.0
    orientation_distance_scale: float = 1.0
    failure_penalty: float = 20.0
    max_penetration: float = .005
    max_linear_speed: float = 5.0
    max_angular_speed: float = 10.0
    task: str = "easy"
    crossing_margin: float = .05
    randomize_object_position: bool = False
    randomize_object_orientation: bool = False
    object_position_range: tuple = (.4, .6)
    object_yaw_range: tuple = (-np.pi, np.pi)

    def __post_init__(self):
        if self.task not in ("easy", "medium"):
            raise ValueError("task must be easy or medium")
        if isinstance(self.num_agents, bool) or not isinstance(self.num_agents, int) or not 1 <= self.num_agents <= 16:
            raise ValueError("num_agents must be an integer from 1 to 16")
        for name in ("randomize_agents", "randomize_object_position", "randomize_object_orientation"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("frame_skip", "max_steps", "success_hold_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, shape in (("object_pose", (3,)), ("agent_positions", (self.num_agents, 2)),
                            ("goal_center", (2,)), ("goal_half_size", (2,)),
                            ("object_position_range", (2,)), ("object_yaw_range", (2,))):
            if name == "agent_positions" and self.agent_positions is None:
                continue
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must have shape {shape} and finite values")
            value = tuple(map(tuple, array.tolist())) if array.ndim == 2 else tuple(array.tolist())
            object.__setattr__(self, name, value)
        if min(self.goal_half_size) <= 0:
            raise ValueError("goal_half_size must be positive")
        if min(self.object_position_range) < 0:
            raise ValueError("object_position_range half-widths cannot be negative")
        if self.randomize_object_position and max(self.object_position_range) <= 0:
            raise ValueError("randomized object position needs a nonzero position range")
        if self.object_yaw_range[0] >= self.object_yaw_range[1]:
            raise ValueError("object_yaw_range must have increasing min/max angles in radians")
        nonnegative = ("success_linear_speed", "success_angular_speed", "distance_weight",
                       "orientation_weight", "progress_weight", "success_reward", "failure_penalty",
                       "distance_scale", "orientation_distance_scale", "max_penetration",
                       "max_linear_speed", "max_angular_speed", "spawn_gap_min", "spawn_gap_max",
                       "crossing_margin")
        for name in (*nonnegative, "goal_yaw"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name in nonnegative and value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, float(value))
        if self.spawn_gap_max <= self.spawn_gap_min:
            raise ValueError("spawn_gap_max must be > spawn_gap_min")
        for name in ("distance_scale", "orientation_distance_scale", "max_penetration",
                     "max_linear_speed", "max_angular_speed"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("model_path must be a nonempty string")

    @property
    def task_id(self):
        return "medium_v1" if self.task == "medium" else "easy_v2"

    @property
    def resolved_model_path(self):
        path = Path(self.model_path).expanduser()
        return path if path.is_absolute() else ROOT / path

    def to_dict(self):
        return asdict(self)


def load_config(config=None, **overrides):
    if isinstance(config, EnvironmentConfig):
        result = config
    else:
        source = DEFAULT_CONFIG_PATH if config is None else config
        values = yaml.safe_load(Path(source).read_text()) if isinstance(source, (str, Path)) else source
        if not isinstance(values, Mapping):
            raise ValueError("environment configuration must be a mapping")
        unknown = set(values) - {f.name for f in fields(EnvironmentConfig)}
        if unknown:
            raise ValueError(f"unsupported environment settings: {sorted(unknown)}")
        result = EnvironmentConfig(**values)
    return replace(result, **{k: v for k, v in overrides.items() if v is not None})
