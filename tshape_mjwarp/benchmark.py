"""Warm-up excluded, synchronized end-to-end throughput; never infers GPU speed."""
from dataclasses import replace
import gc
import time
import torch

from .environment import BatchedEnvironment
from .mappo import MAPPO
from .rollout import RolloutBuffer
from .trainer import collect


def benchmark(config, simulation, learning, counts, warmup=10, steps=100, with_updates=False):
    if warmup < 0 or steps < 1:
        raise ValueError("warmup must be nonnegative and steps positive")
    results = []
    for count in counts:
        sim = replace(simulation, num_envs=count)
        env = None
        buffer = None
        try:
            torch.manual_seed(learning.seed)
            env = BatchedEnvironment(config, sim, learning.seed)
            agent = MAPPO(**env.dimensions, config=learning)
            generator = torch.Generator(device=sim.device).manual_seed(learning.seed)
            def sync():
                if env.device.type == "cuda":
                    torch.cuda.synchronize(env.device)
            for _ in range(warmup):
                actions, _, _ = agent.act(env.observations)
                _, _, _, term, trunc, _ = env.step(actions)
                env.reset(term | trunc)
            sync()
            if env.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(env.device)
            started = time.perf_counter()
            remaining = steps
            while remaining:
                length = min(learning.rollout_steps, remaining)
                if with_updates:
                    buffer = RolloutBuffer(length, count, env.dimensions, env.device)
                    collect(agent, env, buffer)
                    agent.update(buffer, generator)
                else:
                    for _ in range(length):
                        actions, _, _ = agent.act(env.observations)
                        _, _, _, term, trunc, _ = env.step(actions)
                        env.reset(term | trunc)
                remaining -= length
            sync()
            elapsed = time.perf_counter()-started
            gpu = env.device.type == "cuda"
            result = dict(backend=sim.backend, device=sim.device, num_envs=count, status="ok",
                          control_steps_per_world=steps, warmup_steps=warmup, wall_seconds=elapsed,
                          includes_ppo=with_updates, environment_transitions_per_second=steps*count/elapsed,
                          nominal_physics_substeps_per_second=steps*count*config.frame_skip/elapsed,
                          torch_peak_allocated_mib=torch.cuda.max_memory_allocated(env.device)/2**20 if gpu else None,
                          device_free_mib=torch.cuda.mem_get_info(env.device)[0]/2**20 if gpu else None)
            results.append(result)
        except Exception as error:
            results.append(dict(backend=sim.backend, device=sim.device, num_envs=count, status="error",
                                error=f"{type(error).__name__}: {error}"))
            # Device-side assertion invalidates the CUDA context: do not continue.
            if "assert" in str(error).lower():
                break
        finally:
            if env is not None:
                env.close()
            env = None
            buffer = None
            if "agent" in locals():
                del agent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results
