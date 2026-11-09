"""Explicit opt-in GPU tests. Opting in without CUDA fails instead of pretending to pass."""
import os
import subprocess
import sys
import unittest
import torch
from tshape_mjwarp.config import environment_config, SimulationConfig


@unittest.skipUnless(os.environ.get("TSHAPE_RUN_GPU_TESTS") == "1", "Run on remote PC with TSHAPE_RUN_GPU_TESTS=1")
class GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU tests explicitly requested, but CUDA is unavailable")

    def test_gpu_short_parity(self):
        from tshape_mjwarp.validation import validate
        result = validate(environment_config(), SimulationConfig(num_envs=2), steps=5)
        self.assertTrue(result["passed"], result)

    def test_gpu_cli_validate_in_fresh_process(self):
        result = subprocess.run(
            [sys.executable, "-m", "tshape_mjwarp.cli", "validate",
             "--backend", "mjwarp", "--device", "cuda:0",
             "--num-envs", "2", "--num-agents", "8"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_gpu_training_and_checkpoint(self):
        from dataclasses import replace
        from contextlib import redirect_stdout
        from pathlib import Path
        import io
        import tempfile
        from tshape_mjwarp.trainer import train
        from tshape_mjwarp.training_config import MAPPOConfig
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            sim = SimulationConfig(num_envs=2)
            cfg = MAPPOConfig(device="cuda:0", total_steps=8, rollout_steps=2, ppo_epochs=1,
                              actor_hidden_size=8, critic_hidden_size=8, video_interval_steps=0,
                              tensorboard=False, eval_episodes=2)
            self.assertEqual(train(Path(directory)/"run", environment_config(max_steps=2), sim, cfg)["environment_steps"], 8)

    def test_gpu_failure_is_local_and_partial_reset_preserves_other_world(self):
        from tshape_mjwarp.environment import BatchedEnvironment
        env = BatchedEnvironment(environment_config(), SimulationConfig(num_envs=2))
        try:
            env.physics.qpos[0, env.task.object_qpos[0]] = torch.nan
            _, _, rewards, term, _, info = env.step(torch.zeros(2, 4, 2, device=env.device))
            self.assertEqual(term.cpu().tolist(), [True, False])
            self.assertEqual(rewards[0].item(), -20.)
            self.assertTrue(torch.isfinite(env.observations).all())
            state = env.physics.qpos[1].clone()
            rng = env.sampler.rng[1].clone()
            env.reset(term)
            torch.testing.assert_close(env.physics.qpos[1], state)
            torch.testing.assert_close(env.sampler.rng[1], rng)
        finally:
            env.close()
