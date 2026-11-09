from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import shutil
import unittest
import torch

from tshape_mjwarp.config import environment_config, SimulationConfig
from tshape_mjwarp.training_config import MAPPOConfig
from tshape_mjwarp.trainer import train
from tshape_mjwarp.checkpoint import loaded
from tshape_mjwarp.evaluation import evaluate
from tshape_mjwarp.benchmark import benchmark


class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "video needs FFmpeg")
    def test_video_records_first_episode(self):
        from tshape_mjwarp.mappo import MAPPO
        sim = SimulationConfig(backend="reference", device="cpu", num_envs=2)
        env = environment_config(max_steps=2)
        cfg = MAPPOConfig(actor_hidden_size=8, critic_hidden_size=8, video_width=64, video_height=64)
        agent = MAPPO(39, 38, num_agents=4, config=cfg)
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate(agent, env, sim, episodes=2, video_dir=directory)
            self.assertTrue((Path(directory)/"policy.mp4").exists())
            self.assertEqual(report["video"]["frames"], 3)

    def test_train_checkpoint_evaluate_and_resume(self):
        sim = SimulationConfig(backend="reference", device="cpu", num_envs=2)
        env = environment_config(max_steps=2, randomize_object_position=True, randomize_object_orientation=True)
        cfg = MAPPOConfig(total_steps=8, rollout_steps=2, ppo_epochs=1, actor_hidden_size=8,
                          critic_hidden_size=8, video_interval_steps=0, tensorboard=False, eval_episodes=2)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            result = train(root/"run", env, sim, cfg)
            self.assertEqual(result["environment_steps"], 8)
            path = root/"run/checkpoints/latest.pt"
            with loaded(path, sim) as (agent, saved, _, payload):
                self.assertTrue(saved.randomize_object_position)
                self.assertEqual(payload["format"], "tshape-mjwarp-mappo-v1")
                a = evaluate(agent, saved, sim, episodes=3, seed=123)
                self.assertEqual(a, evaluate(agent, saved, sim, episodes=3, seed=123))
                self.assertEqual([r["seed"] for r in a["reports"]], [123, 124, 125])
            result = train(root/"resume", env, sim, replace(cfg, total_steps=12), path)
            self.assertEqual(result["environment_steps"], 12)
            with self.assertRaises(FileExistsError):
                train(root/"run", env, sim, cfg)
            with self.assertRaises(ValueError):
                train(root/"badbudget", env, sim, replace(cfg, total_steps=3))
            with self.assertRaises(ValueError):
                train(root/"badresume", env, sim, replace(cfg, total_steps=12, gamma=.5), path)

    def test_benchmark_reports_backend_and_units(self):
        sim = SimulationConfig(backend="reference", device="cpu", num_envs=1)
        cfg = MAPPOConfig(rollout_steps=1, ppo_epochs=1, actor_hidden_size=8, critic_hidden_size=8)
        result = benchmark(environment_config(max_steps=2), sim, cfg, [1, 2], warmup=0, steps=2, with_updates=True)
        self.assertTrue(all(row["status"] == "ok" for row in result), result)
        self.assertTrue(all(row["backend"] == "reference" and row["includes_ppo"] for row in result))
