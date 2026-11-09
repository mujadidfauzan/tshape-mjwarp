import unittest
from unittest.mock import patch

import torch

from tshape_mjwarp.config import environment_config, SimulationConfig
from tshape_mjwarp.environment import BatchedEnvironment
from tshape_mjwarp.reference.envs.tshape_env import TShapeParallelEnv
from tshape_mjwarp.validation import validate


class ValidationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = environment_config(randomize_agents=False)
        self.sim = SimulationConfig(backend="reference", device="cpu", num_envs=2)

    def test_invalid_steps_rejected_before_environment_creation(self):
        with patch("tshape_mjwarp.validation.BatchedEnvironment") as constructor:
            for steps in (0, -1, True, 1.5):
                with self.subTest(steps=steps), self.assertRaises(ValueError):
                    validate(self.cfg, self.sim, steps=steps)
            constructor.assert_not_called()

    def test_cpu_parity_checks_healthy_steps(self):
        result = validate(self.cfg, self.sim, steps=3)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["batched_stable"])
        self.assertTrue(result["reference_stable"])
        self.assertEqual(result["controls_checked"], 3)

    def test_invalid_tolerance_rejected_before_environment_creation(self):
        with patch("tshape_mjwarp.validation.BatchedEnvironment") as constructor:
            for tolerance in (0., -1., float("nan"), float("inf")):
                with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                    validate(self.cfg, self.sim, tolerance=tolerance)
            constructor.assert_not_called()

    def test_non_finite_trajectory_error_fails_validation(self):
        original_step = TShapeParallelEnv.step
        def non_finite_step(reference, actions):
            result = original_step(reference, actions)
            reference.physics.data.qpos[0] = float("nan")
            return result

        with patch.object(TShapeParallelEnv, "step", non_finite_step):
            result = validate(self.cfg, self.sim, steps=1)
        self.assertIs(result["passed"], False)

    def test_failure_in_second_world_fails_validation(self):
        env = BatchedEnvironment(self.cfg, self.sim)
        original_step = env.step
        def failed_step(actions):
            result = original_step(actions)
            result[5]["stable"][1] = False
            result[3][1] = True
            return result

        with patch("tshape_mjwarp.validation.BatchedEnvironment", return_value=env), \
             patch.object(env, "step", side_effect=failed_step):
            result = validate(self.cfg, self.sim, steps=3)
        self.assertFalse(result["passed"], result)
        self.assertFalse(result["batched_stable"])
        self.assertTrue(result["reference_stable"])
        self.assertEqual(result["controls_checked"], 1)

    def test_reference_failure_fails_validation_despite_matching_positions(self):
        original_step = TShapeParallelEnv.step
        def failed_step(reference, actions):
            result = original_step(reference, actions)
            for info in result[4].values():
                info["stable"] = False
            return result

        with patch.object(TShapeParallelEnv, "step", failed_step):
            result = validate(self.cfg, self.sim, steps=3)
        self.assertFalse(result["passed"], result)
        self.assertTrue(result["batched_stable"])
        self.assertFalse(result["reference_stable"])
        self.assertLess(result["trajectory_max_abs_qpos_error"], result["tolerance"])

    def test_other_world_timeout_stops_before_next_step(self):
        env = BatchedEnvironment(self.cfg, self.sim)
        original_step = env.step
        def timed_out_step(actions):
            result = original_step(actions)
            result[4][1] = True
            env.done[1] = True
            return result

        with patch("tshape_mjwarp.validation.BatchedEnvironment", return_value=env), \
             patch.object(env, "step", side_effect=timed_out_step):
            result = validate(self.cfg, self.sim, steps=3)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["controls_checked"], 1)
