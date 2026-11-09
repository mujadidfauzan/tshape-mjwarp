from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import torch

from tshape_mjwarp.physics import WarpPhysics


class PhysicsTests(unittest.TestCase):
    def test_non_finite_substep_preserves_path_and_healthy_world(self):
        # Execute the real rollback/metric loop with a controlled failing physics step.
        for invalid_position in (float("nan"), float("inf")):
            with self.subTest(invalid_position=invalid_position):
                physics = WarpPhysics.__new__(WarpPhysics)
                physics.config = SimpleNamespace(frame_skip=3)
                physics.qpos = torch.zeros(2, 3)
                physics.qvel = torch.zeros_like(physics.qpos)
                physics.ctrl = torch.zeros(2, 2)
                physics.time = torch.zeros(2)
                physics.safe_qpos, physics.safe_qvel = physics.qpos.clone(), physics.qvel.clone()
                physics.previous_pairs = torch.zeros(2, 1, dtype=torch.bool)
                physics.cpu = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=.001)))
                physics.model = object()
                physics.data = SimpleNamespace(qacc_warmstart=torch.zeros_like(physics.qpos))
                physics.wp = SimpleNamespace(to_torch=lambda values: values)
                physics.scope = nullcontext
                physics.contact_state = lambda: None
                physics.task = SimpleNamespace(
                    motor_limits=torch.tensor([[[-3., 3.], [-3., 3.]]]),
                    motor_ids=torch.tensor([[0, 1]]),
                    geometry=lambda q, v: (None, None, q[:, :2]),
                )
                calls = 0
                def step(model, data):
                    nonlocal calls
                    calls += 1
                    physics.qpos[:, 0] += .1
                    physics.time += .001
                    if calls >= 2:
                        physics.qpos[0, 0] = invalid_position

                physics.mjw = SimpleNamespace(step=step, forward=lambda *args: None)
                physics.diagnostics = lambda: (
                    (~torch.isfinite(physics.qpos).all(1)).long(),
                    torch.zeros(2), torch.zeros(2, 1, dtype=torch.bool),
                )
                result = physics.step(torch.zeros(2, 1, 2))
                self.assertEqual(result["failed"].tolist(), [True, False])
                torch.testing.assert_close(result["path_length"], torch.tensor([.1, .3]))
                torch.testing.assert_close(physics.qpos[:, 0], torch.tensor([.1, .3]))
                torch.testing.assert_close(physics.time, torch.tensor([.001, .003]))
