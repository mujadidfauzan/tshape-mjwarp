"""Optional real MJWarp CPU diagnostic; never a GPU performance measurement."""
import os
import unittest
from tshape_mjwarp.config import environment_config, SimulationConfig


@unittest.skipUnless(os.environ.get("TSHAPE_RUN_WARP_CPU_TESTS") == "1", "Opt in to MJWarp CPU compilation")
class WarpCpuTests(unittest.TestCase):
    def test_real_mjwarp_short_parity(self):
        from tshape_mjwarp.validation import validate
        result = validate(environment_config(), SimulationConfig(device="cpu", num_envs=2), steps=3)
        self.assertTrue(result["passed"], result)

    def test_real_mjwarp_cpu_training_smoke(self):
        from pathlib import Path
        from contextlib import redirect_stdout
        import io
        import tempfile
        from tshape_mjwarp.trainer import train
        from tshape_mjwarp.training_config import MAPPOConfig
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            cfg = MAPPOConfig(device="cpu", total_steps=4, rollout_steps=1, ppo_epochs=1,
                              actor_hidden_size=8, critic_hidden_size=8, video_interval_steps=0,
                              tensorboard=False, eval_episodes=1)
            result = train(Path(directory)/"run", environment_config(max_steps=2),
                           SimulationConfig(device="cpu", num_envs=2), cfg)
            self.assertEqual(result["environment_steps"], 4)
