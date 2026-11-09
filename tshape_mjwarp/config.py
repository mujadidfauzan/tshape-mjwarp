"""Independent configuration; importing this project never modifies the baseline."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import yaml

from .reference.envs.config import EnvironmentConfig, load_config as load_environment
from .training_config import load_mappo_config

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SimulationConfig:
    backend: str = "mjwarp"
    device: str = "cuda:0"
    num_envs: int = 32
    nconmax: int = 256
    njmax: int = 1024

    def __post_init__(self):
        if self.backend not in ("reference", "mjwarp"):
            raise ValueError("backend must be reference or mjwarp")
        if self.device != "cpu" and self.device != "cuda" and not (
                self.device.startswith("cuda:") and self.device[5:].isdigit()):
            raise ValueError("device must be cpu or cuda:<index>")
        if self.backend == "reference" and self.device != "cpu":
            raise ValueError("reference backend is a CPU correctness oracle, not GPU acceleration")
        for name in ("num_envs", "nconmax", "njmax"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self):
        return asdict(self)


def environment_config(path=None, **overrides):
    cfg = load_environment(path or ROOT/"configs/environment.yaml", **overrides)
    if cfg.task != "medium":
        raise ValueError("initial MJWarp implementation supports the Medium doorway task only")
    return cfg


def training_config(path=None, **overrides):
    values = yaml.safe_load(Path(path or ROOT/"configs/training.yaml").read_text())
    if set(values) != {"simulation", "learning"}:
        raise ValueError("training YAML needs simulation and learning sections")
    sim = SimulationConfig(**values["simulation"])
    sim = replace(sim, **{k: v for k, v in overrides.items() if k in asdict(sim) and v is not None})
    learning = load_mappo_config(values["learning"], device=sim.device,
                                 **{k: v for k, v in overrides.items()
                                    if k not in asdict(sim) and v is not None})
    return sim, learning
