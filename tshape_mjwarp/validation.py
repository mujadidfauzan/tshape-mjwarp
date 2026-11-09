"""Short physics/feature comparison against an independent C MuJoCo oracle."""
import numpy as np
import torch
import mujoco

from .environment import BatchedEnvironment
from .reference.envs.tshape_env import TShapeParallelEnv


def validate(config, simulation, steps=5, tolerance=.02):
    env = BatchedEnvironment(config, simulation)
    reference = TShapeParallelEnv(config.to_dict())
    try:
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
        for _ in range(steps):
            action = rng.uniform(-.1, .1, (simulation.num_envs, config.num_agents, 2)).astype(np.float32)
            result = env.step(torch.as_tensor(action, device=env.device))
            reference.step(dict(zip(names, action[0])))
            errors.append(float(np.max(np.abs(reference.physics.data.qpos-env.physics.qpos[0].cpu().numpy()))))
            if result[3][0] or result[4][0]:
                break
        passed = feature_error < 1e-4 and state_error < 1e-4 and max(errors, default=0.) <= tolerance
        return dict(backend=simulation.backend, device=simulation.device, passed=passed,
                    initial_observation_max_abs_error=feature_error, initial_state_max_abs_error=state_error,
                    trajectory_max_abs_qpos_error=max(errors, default=0.), tolerance=tolerance,
                    controls_checked=len(errors), note="Short numerical check, not bitwise equivalence or a learning benchmark")
    finally:
        env.close()
        reference.close()
