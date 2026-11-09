"""Short physics/feature comparison against an independent C MuJoCo oracle."""
import numpy as np
import torch
import mujoco

from .environment import BatchedEnvironment
from .reference.envs.tshape_env import TShapeParallelEnv


def validate(config, simulation, steps=5, tolerance=.02):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("validation steps must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("validation tolerance must be finite and positive")
    env = BatchedEnvironment(config, simulation)
    reference = None
    try:
        reference = TShapeParallelEnv(config.to_dict())
        reference.reset(seed=42)
        reference.physics.data.qpos[:] = env.physics.qpos[0].cpu().numpy()
        mujoco.mj_forward(reference.physics.model, reference.physics.data)
        names = reference.possible_agents
        expected_obs = np.stack([reference._observations()[n] for n in names])
        feature_error = float(np.max(np.abs(expected_obs-env.observations[0].cpu().numpy())))
        state_error = float(np.max(np.abs(reference.state()-env.states[0].cpu().numpy())))
        reference._last_observations = reference._observations()
        reference._last_state = reference.state()
        rng = np.random.default_rng(123)
        errors = []
        batched_stable = reference_stable = True
        for _ in range(steps):
            action = rng.uniform(-.1, .1, (simulation.num_envs, config.num_agents, 2)).astype(np.float32)
            result = env.step(torch.as_tensor(action, device=env.device))
            ref_result = reference.step(dict(zip(names, action[0])))
            batched_stable &= bool(result[5]["stable"].all().item())
            reference_stable &= bool(ref_result[4][names[0]]["stable"])
            errors.append(float(np.max(np.abs(reference.physics.data.qpos-env.physics.qpos[0].cpu().numpy()))))
            if (not batched_stable or not reference_stable
                    or bool((result[3] | result[4]).any().item())
                    or any(ref_result[2].values()) or any(ref_result[3].values())):
                break
        trajectory_error = max(errors)
        passed = (batched_stable and reference_stable
                  and bool(np.isfinite([feature_error, state_error, *errors]).all())
                  and feature_error < 1e-4 and state_error < 1e-4 and trajectory_error <= tolerance)
        return dict(backend=simulation.backend, device=simulation.device, passed=passed,
                    batched_stable=batched_stable, reference_stable=reference_stable,
                    initial_observation_max_abs_error=feature_error, initial_state_max_abs_error=state_error,
                    trajectory_max_abs_qpos_error=trajectory_error, tolerance=tolerance,
                    controls_checked=len(errors), note="Short numerical check, not bitwise equivalence or a learning benchmark")
    finally:
        env.close()
        if reference is not None:
            reference.close()
