"""Seeded held-out evaluation. Video renders the actual simulated GPU states."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
import mujoco

from .environment import BatchedEnvironment
from .artifacts import nullable


def evaluate(agent, config, simulation, episodes=20, seed=2000000, video_dir=None, render=False):
    if episodes < 1 or seed < 0:
        raise ValueError("episodes must be positive and seed nonnegative")
    sim = replace(simulation, num_envs=min(simulation.num_envs, episodes))
    env = BatchedEnvironment(config, sim, seed)
    if env.dimensions != agent.dimensions:
        raise ValueError("policy dimensions differ from environment")
    e = sim.num_envs
    ids = torch.arange(e, device=env.device)
    env.sampler.rng.copy_(seed+ids)
    env.reset()
    next_id, reports = e, []
    with ExitStack() as stack:
        stack.callback(env.close)
        video = viewer = None
        if video_dir or render:
            mjm, mjd = env.oracle.model, mujoco.MjData(env.oracle.model)
            # Display the full destination side, as in the CPU reference environment.
            marker = mjm.site("goal")
            exit_x = env.task.door.x_max+config.crossing_margin
            marker.pos[:2] = [(exit_x+env.oracle.upper[0])/2, 0]
            marker.size[:2] = [(env.oracle.upper[0]-exit_x)/2, env.oracle.upper[1]]
            if video_dir:
                from .video import EpisodeVideo
                video = stack.enter_context(EpisodeVideo(SimpleNamespace(model=mjm, data=mjd,
                                control_dt=env.oracle.control_dt), Path(video_dir)/"policy.mp4",
                                agent.config.video_width, agent.config.video_height, agent.config.video_crf))
            if render:
                from mujoco import viewer as viewer_module
                viewer = stack.enter_context(viewer_module.launch_passive(mjm, mjd))
                viewer.cam.lookat[:] = [0, 0, 0]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 8, 90, -90
            def display():
                mjd.qpos[:] = env.physics.qpos[0].cpu().numpy()
                mjd.qvel[:] = env.physics.qvel[0].cpu().numpy()
                mujoco.mj_forward(mjm, mjd)
                if video is not None:
                    video.capture()
                if viewer is not None:
                    viewer.sync()
            display()
        recording = bool(video_dir or render)
        while len(reports) < episodes:
            started = time.perf_counter()
            actions, _, _ = agent.act(env.observations, deterministic=True)
            _, _, _, term, trunc, info = env.step(actions)
            if recording:
                display()
            done = term | trunc
            completed = done.nonzero().flatten().cpu().tolist()
            for i in completed:
                episode_id = int(ids[i])
                if episode_id < episodes:
                    row = {key: (value[i].cpu().tolist()) for key, value in info.items()}
                    if not row["success"]:
                        row["completion_time_s"] = None
                    row.update(seed=seed+episode_id, episode=episode_id+1,
                               end_reason="success" if row["success"] else "physics_failure" if not row["stable"] else "time_limit",
                               initial_qpos=env.initial_qpos[i].cpu().tolist())
                    reports.append(nullable(row))
                ids[i] = next_id
                next_id += 1
            if recording and 0 in completed:
                recording = False
            if len(reports) >= episodes:
                break
            env.sampler.rng.copy_(torch.where(done, seed+ids, env.sampler.rng))
            env.reset(done)
            if viewer is not None:
                if not viewer.is_running():
                    raise InterruptedError("Viewer closed before evaluation completed")
                time.sleep(max(0., env.oracle.control_dt-(time.perf_counter()-started)))
        reports.sort(key=lambda row: row["episode"])
        times = [r["completion_time_s"] for r in reports if r["success"]]
        summary = dict(episodes=len(reports), success_rate=float(np.mean([r["success"] for r in reports])),
                       stable_rate=float(np.mean([r["stable"] for r in reports])),
                       mean_return=float(np.mean([r["episode_return"] for r in reports])),
                       mean_completion_time_s=float(np.mean(times)) if times else None)
        return dict(backend=sim.backend, reports=reports, summary=summary,
                    video=video.metadata() if video is not None else None)
