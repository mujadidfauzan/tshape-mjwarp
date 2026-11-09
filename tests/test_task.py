import unittest
from unittest.mock import patch
from itertools import product
import numpy as np
import torch
import mujoco

from tshape_mjwarp.config import environment_config, SimulationConfig
from tshape_mjwarp.environment import BatchedEnvironment
from tshape_mjwarp.reference.envs.tshape_env import TShapeParallelEnv


class TaskTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = environment_config(max_steps=2)
        self.sim = SimulationConfig(backend="reference", device="cpu", num_envs=3)

    def test_observations_and_route_match_reference(self):
        env = BatchedEnvironment(self.cfg, self.sim)
        ref = TShapeParallelEnv(self.cfg.to_dict())
        ref.reset()
        ref.physics.data.qpos[:] = env.physics.qpos[0].numpy()
        mujoco.mj_forward(ref.physics.model, ref.physics.data)
        np.testing.assert_allclose(env.observations[0], np.stack(list(ref._observations().values())), atol=1e-6)
        np.testing.assert_allclose(env.states[0], ref.state(), atol=1e-6)
        points = torch.tensor([[-1.5, 2.], [-.1, .2], [.2, .1], [-1.5, 0.], [0., 1.]])
        expected = [env.oracle.doorway.route_distance(p.numpy(), self.cfg.goal_center) for p in points]
        np.testing.assert_allclose(env.task.distance(points), expected, atol=1e-6)

    def test_spawn_independent_flags_safety_and_masked_rng(self):
        for agents, xy, yaw in product((False, True), repeat=3):
            cfg = environment_config(num_agents=8, randomize_agents=agents,
                                     randomize_object_position=xy, randomize_object_orientation=yaw)
            a = BatchedEnvironment(cfg, self.sim, 42)
            b = BatchedEnvironment(cfg, self.sim, 42)
            torch.testing.assert_close(a.physics.qpos, b.physics.qpos)
            q, rng = a.physics.qpos.clone(), a.sampler.rng.clone()
            a.reset(torch.tensor([True, False, False]))
            torch.testing.assert_close(a.physics.qpos[1:], q[1:])
            torch.testing.assert_close(a.sampler.rng[1:], rng[1:])
            self.assertFalse(a.task.boundary_penetration(a.physics.qpos, a.physics.qvel).any())
            for world in a.physics.worlds:
                self.assertTrue(world.diagnostics()["stable"])

    def test_timeout_bootstrap_state_and_partial_reset(self):
        env = BatchedEnvironment(self.cfg, self.sim)
        for _ in range(2):
            obs, state, reward, term, trunc, info = env.step(torch.zeros(3, 4, 2))
        self.assertFalse(term.any())
        self.assertTrue(trunc.all())
        self.assertTrue((reward > 0).all())
        final_state = state.clone()
        env.reset(env.done)
        self.assertFalse(torch.equal(final_state, env.states))
        self.assertFalse(env.steps.any())

    def test_empty_reset_skips_sampling_and_physics(self):
        env = BatchedEnvironment(self.cfg, self.sim)
        try:
            env.step(torch.zeros(3, 4, 2))
            before = [value.clone() for value in (env.physics.qpos, env.physics.qvel,
                      env.observations, env.states, env.steps, env.returns, env.sampler.rng)]
            with patch.object(env.sampler, "sample") as sample, patch.object(env.physics, "reset") as reset:
                observations, states = env.reset(torch.zeros(3, dtype=torch.bool))
                sample.assert_not_called()
                reset.assert_not_called()
            self.assertIs(observations, env.observations)
            self.assertIs(states, env.states)
            after = (env.physics.qpos, env.physics.qvel, env.observations, env.states,
                     env.steps, env.returns, env.sampler.rng)
            for expected, actual in zip(before, after):
                torch.testing.assert_close(actual, expected)
        finally:
            env.close()

    def test_success_uses_whole_footprint_and_failure_wins(self):
        env = BatchedEnvironment(self.cfg, self.sim)
        q = env.physics.qpos.clone()
        q[:, env.task.object_qpos[:2]] = torch.tensor([[.5, 0.], [1.2, 1.2], [1.2, 1.2]])-env.task.object_origin
        q[:, env.task.object_qpos[2]] = torch.pi/2
        env.physics.reset(torch.ones(3, dtype=torch.bool), q)
        env.physics.worlds[2].data.qvel[env.oracle.object_dofs[0]] = 6.
        _, _, reward, term, trunc, info = env.step(torch.zeros(3, 4, 2))
        self.assertEqual(info["success"].tolist(), [False, True, False])
        self.assertEqual(term.tolist(), [False, True, True])
        self.assertEqual(reward[2], -20.)
