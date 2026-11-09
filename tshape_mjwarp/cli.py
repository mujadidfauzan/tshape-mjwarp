"""Run with python -m tshape_mjwarp.cli from this standalone project root."""
import argparse
from dataclasses import replace
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "validate", "benchmark", "train", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--config", type=Path)
        p.add_argument("--env-config", type=Path)
        p.add_argument("--backend", choices=("mjwarp", "reference"))
        p.add_argument("--device")
        p.add_argument("--num-envs", type=int)
        p.add_argument("--num-agents", type=int)
        p.add_argument("--json", type=Path)
        p.add_argument("--nconmax", type=int)
        p.add_argument("--njmax", type=int)
        if name in ("train", "benchmark", "validate"):
            for flag in ("randomize-agents", "randomize-object-position", "randomize-object-orientation"):
                p.add_argument("--"+flag, action=argparse.BooleanOptionalAction, default=None)
        if name == "train":
            p.add_argument("--run-dir", type=Path, required=True)
            p.add_argument("--total-steps", type=int)
            p.add_argument("--resume", type=Path)
        if name == "evaluate":
            p.add_argument("checkpoint", type=Path)
            p.add_argument("--episodes", type=int, default=20)
            p.add_argument("--seed", type=int, default=2000000)
            p.add_argument("--video-dir", type=Path)
            p.add_argument("--render", action="store_true")
        if name == "benchmark":
            p.add_argument("--counts", nargs="+", type=int, default=[16, 32, 64, 128])
            p.add_argument("--warmup", type=int, default=10)
            p.add_argument("--steps", type=int, default=100)
            p.add_argument("--with-updates", action="store_true")
        if name == "validate":
            p.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if not getattr(args, "render", False) and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    from .config import environment_config, training_config, SimulationConfig
    from .artifacts import write_json
    sim_overrides = {name: getattr(args, name) for name in ("backend", "device", "num_envs", "nconmax", "njmax")
                     if getattr(args, name) is not None}
    try:
        if args.command == "evaluate":
            from .checkpoint import loaded, read
            from .evaluation import evaluate
            if args.env_config or args.config or args.num_agents:
                parser.error("evaluation restores geometry, task and agent count from the checkpoint")
            payload = read(args.checkpoint)
            sim = replace(SimulationConfig(**payload["simulation"]), **sim_overrides)
            torch.set_num_threads(1)
            with loaded(args.checkpoint, sim) as (agent, config, sim, payload):
                result = evaluate(agent, config, sim, args.episodes, args.seed, args.video_dir, args.render)
                result.update(checkpoint=str(args.checkpoint.resolve()), model_sha256=payload["model_sha256"])
        else:
            sim, learning = training_config(args.config, **sim_overrides)
            overrides = {name: getattr(args, name, None) for name in
                         ("num_agents", "randomize_agents", "randomize_object_position", "randomize_object_orientation")}
            config = environment_config(args.env_config, **overrides)
            if args.command == "check":
                packages = {}
                for package in ("torch", "mujoco", "mujoco-warp", "warp-lang"):
                    try:
                        packages[package] = version(package)
                    except PackageNotFoundError:
                        packages[package] = None
                compatible = (packages["mujoco"] == "3.5.0" and packages["mujoco-warp"] == "3.5.0"
                              and packages["warp-lang"] == "1.11.0")
                result = dict(packages=packages, cuda_available=torch.cuda.is_available(),
                              requested_device=sim.device, dependencies_compatible=compatible)
                if torch.cuda.is_available():
                    device = torch.device(sim.device if sim.device.startswith("cuda") else "cuda:0")
                    props = torch.cuda.get_device_properties(device)
                    result.update(gpu=props.name, vram_mib=props.total_memory/2**20)
                result["ready"] = compatible and (sim.device == "cpu" or result["cuda_available"])
            elif args.command == "validate":
                from .validation import validate
                result = validate(config, sim, args.steps)
            elif args.command == "benchmark":
                from .benchmark import benchmark
                result = dict(results=benchmark(config, sim, learning, args.counts, args.warmup, args.steps, args.with_updates))
            else:
                from .trainer import train
                if args.resume:
                    if args.total_steps is None:
                        parser.error("resume requires a larger cumulative --total-steps")
                    if args.config or args.env_config or any(v is not None for v in overrides.values()) or sim_overrides:
                        parser.error("resume restores simulation, task, randomization and optimizer settings")
                    from .training_config import load_mappo_config
                    from .checkpoint import read
                    payload = read(args.resume)
                    sim = SimulationConfig(**payload["simulation"])
                    learning = replace(load_mappo_config(payload["learning"]), total_steps=args.total_steps)
                elif args.total_steps is not None:
                    learning = replace(learning, total_steps=args.total_steps)
                result = train(args.run_dir, config, sim, learning, args.resume)
        if args.json:
            write_json(args.json, result)
        print(json.dumps(result if args.command != "evaluate" else result["summary"], indent=2, allow_nan=False))
        if args.command == "check" and not result["ready"]:
            return 2
        if args.command == "validate" and not result["passed"]:
            return 1
        if args.command == "benchmark" and any(r["status"] != "ok" for r in result["results"]):
            return 1
        return 0
    except (ValueError, RuntimeError, FileExistsError, ImportError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
