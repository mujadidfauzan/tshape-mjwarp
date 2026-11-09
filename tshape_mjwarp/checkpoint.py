"""Self-contained checkpoints, distinct from the baseline checkpoint format."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import torch

from .config import environment_config, SimulationConfig
from .mappo import MAPPO
from .training_config import load_mappo_config

FORMAT = "tshape-mjwarp-mappo-v1"


def save(path, agent, environment, simulation, steps, updates, best_score, generator, evaluations=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = environment.oracle.model_xml
    payload = dict(format=FORMAT, dimensions=agent.dimensions, learning=agent.config.to_dict(),
                   simulation=simulation.to_dict(), environment=environment.config.to_dict(),
                   model_xml=xml, model_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                   model=agent.state_dict(), actor_optimizer=agent.actor_optimizer.state_dict(),
                   critic_optimizer=agent.critic_optimizer.state_dict(), steps=steps, updates=updates,
                   evaluations=evaluations,
                   best_score=list(best_score), spawn_rng=environment.sampler.rng,
                   minibatch_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state(agent.device) if agent.device.type == "cuda" else None)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def read(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("not a standalone tshape-mjwarp checkpoint; baseline files are not imported implicitly")
    if hashlib.sha256(payload["model_xml"].encode()).hexdigest() != payload["model_sha256"]:
        raise ValueError("checkpoint XML hash mismatch")
    return payload


@contextmanager
def loaded(path, simulation=None, optimizers=False):
    payload = read(path)
    sim = simulation or SimulationConfig(**payload["simulation"])
    cfg = replace(load_mappo_config(payload["learning"]), device=sim.device)
    agent = MAPPO(**payload["dimensions"], config=cfg)
    agent.load_state_dict(payload["model"])
    if optimizers:
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    with tempfile.TemporaryDirectory(prefix="tshape-mjwarp-checkpoint-") as directory:
        model = Path(directory)/"arena.xml"
        model.write_text(payload["model_xml"])
        env = environment_config(payload["environment"], model_path=str(model))
        yield agent, env, sim, payload
