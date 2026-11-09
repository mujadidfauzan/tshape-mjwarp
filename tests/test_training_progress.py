from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from tshape_mjwarp import checkpoint
from tshape_mjwarp.config import environment_config, SimulationConfig, training_config
from tshape_mjwarp.evaluation import evaluate
from tshape_mjwarp.mappo import MAPPO
from tshape_mjwarp.trainer import collect, format_duration, train
from tshape_mjwarp.training_config import MAPPOConfig


class TrainingProgressTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.sim = SimulationConfig(backend="reference", device="cpu", num_envs=2)
        self.env = environment_config(max_steps=1, randomize_agents=False)
        self.cfg = MAPPOConfig(total_steps=8, rollout_steps=1, ppo_epochs=1,
                               actor_hidden_size=8, critic_hidden_size=8,
                               video_interval_steps=0, tensorboard=False, eval_episodes=1)

    def test_video_interval_validation_and_yaml_defaults(self):
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(self.cfg, video_interval_evaluations=invalid)
        _, cfg = training_config()
        self.assertEqual(cfg.video_interval_evaluations, 5)
        self.assertEqual((cfg.video_width, cfg.video_height), (1920, 1080))
        self.assertEqual(self.cfg.video_interval_evaluations, 0)

    def test_eta_includes_evaluation_checkpoint_time_and_resumes_from_saved_steps(self):
        clock = [0.]
        original_update, original_save = MAPPO.update, checkpoint.save
        def timed_collect(*args):
            result = collect(*args)
            clock[0] += 2.
            return result

        def timed_update(agent, *args):
            result = original_update(agent, *args)
            clock[0] += 1.
            return result

        def timed_evaluate(*args, **kwargs):
            clock[0] += 5.
            return dict(summary=dict(success_rate=0., mean_completion_time_s=None, mean_return=0.),
                        video=None, reports=[])

        def timed_save(*args, **kwargs):
            original_save(*args, **kwargs)
            clock[0] += 2.

        cfg = replace(self.cfg, eval_interval_steps=4, checkpoint_interval_steps=4)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output), \
             patch("tshape_mjwarp.trainer.time.perf_counter", side_effect=lambda: clock[0]), \
             patch("tshape_mjwarp.trainer.collect", side_effect=timed_collect), \
             patch.object(MAPPO, "update", timed_update), \
             patch("tshape_mjwarp.trainer.evaluate", side_effect=timed_evaluate), \
             patch("tshape_mjwarp.trainer.checkpoint.save", side_effect=timed_save):
            root = Path(directory)
            train(root/"run", self.env, self.sim, cfg)
            rows = [json.loads(line) for line in (root/"run/updates.jsonl").read_text().splitlines()]
            self.assertEqual([row["wall_seconds"] for row in rows], [3., 15., 18., 28.])
            self.assertEqual([row["eta_seconds"] for row in rows], [9., 15., 6., 0.])
            train(root/"resume", self.env, self.sim, replace(cfg, total_steps=12),
                  root/"run/checkpoints/latest.pt")
            resumed = [json.loads(line) for line in (root/"resume/updates.jsonl").read_text().splitlines()]
            self.assertEqual(resumed[0]["environment_steps"], 10)
            self.assertEqual(resumed[0]["eta_seconds"], 3.)
            self.assertEqual(resumed[-1]["eta_seconds"], 0.)
        self.assertIn("ETA=00:00:09", output.getvalue())
        self.assertEqual(format_duration(90061), "25:01:01")

    def test_missing_ffmpeg_fails_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("tshape_mjwarp.trainer.shutil.which", return_value=None):
            root = Path(directory)/"run"
            with self.assertRaisesRegex(RuntimeError, "FFmpeg"):
                train(root, self.env, self.sim, replace(self.cfg, video_interval_evaluations=5))
            self.assertFalse(root.exists())

    def test_checkpoint_without_new_fields_still_resumes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            train(root/"run", self.env, self.sim, self.cfg)
            payload = checkpoint.read(root/"run/checkpoints/latest.pt")
            payload.pop("evaluations")
            payload["learning"].pop("video_interval_evaluations")
            torch.save(payload, root/"legacy.pt")
            train(root/"resume", self.env, self.sim, replace(self.cfg, total_steps=12), root/"legacy.pt")
            resumed = checkpoint.read(root/"resume/checkpoints/latest.pt")
            self.assertEqual(resumed["steps"], 12)
            self.assertEqual(resumed["evaluations"], 1)
            self.assertEqual(resumed["learning"]["video_interval_evaluations"], 0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "automatic videos need FFmpeg")
    def test_video_every_two_evaluations_continues_after_resume(self):
        cfg = replace(self.cfg, total_steps=6, eval_interval_steps=2, checkpoint_interval_steps=2,
                      video_interval_evaluations=2, video_width=64, video_height=64)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            train(root/"run", self.env, self.sim, cfg)
            path = root/"run/checkpoints/latest.pt"
            self.assertEqual(checkpoint.read(path)["evaluations"], 3)
            train(root/"resume", self.env, self.sim, replace(cfg, total_steps=12), path)
            for name, numbers in (("run", [1, 2, 3]), ("resume", [4, 5, 6])):
                reports = [json.loads(line) for line in (root/name/"evaluations.jsonl").read_text().splitlines()]
                self.assertEqual([row["evaluation"] for row in reports], numbers)
                for row in reports:
                    if row["evaluation"] % 2:
                        self.assertIsNone(row["video"])
                    else:
                        video = row["video"]
                        self.assertTrue(Path(video["path"]).is_file())
                        self.assertEqual(video["frames"], 2)
                        self.assertIn(f"eval_{row['evaluation']:05d}_step_", video["path"])
            self.assertEqual(len(list(root.rglob("policy.mp4"))), 3)
            self.assertEqual(checkpoint.read(root/"resume/checkpoints/latest.pt")["evaluations"], 6)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "HD video check needs FFmpeg/ffprobe")
    def test_full_hd_video_has_requested_dimensions(self):
        agent = MAPPO(39, 38, num_agents=4, config=self.cfg)
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate(agent, self.env, self.sim, episodes=1, video_dir=directory)
            result = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,codec_name,nb_frames",
                "-of", "json", str(Path(directory)/"policy.mp4"),
            ], check=True, capture_output=True, text=True)
            stream = json.loads(result.stdout)["streams"][0]
            self.assertEqual((stream["width"], stream["height"]), (1920, 1080))
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual(int(stream["nb_frames"]), report["video"]["frames"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "CLI video check needs FFmpeg")
    def test_cli_video_override_works_for_new_run_and_resume(self):
        cfg = replace(self.cfg, total_steps=4, eval_interval_steps=2, checkpoint_interval_steps=2,
                      video_width=64, video_height=64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"training.yaml").write_text(yaml.safe_dump(dict(
                simulation=self.sim.to_dict(), learning=cfg.to_dict())))
            (root/"environment.yaml").write_text(yaml.safe_dump(self.env.to_dict()))
            base = [sys.executable, "-m", "tshape_mjwarp.cli", "train"]
            first = subprocess.run(base + [
                "--config", str(root/"training.yaml"), "--env-config", str(root/"environment.yaml"),
                "--video-interval-evaluations", "2", "--run-dir", str(root/"run"),
            ], capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            latest = root/"run/checkpoints/latest.pt"
            self.assertEqual(checkpoint.read(latest)["learning"]["video_interval_evaluations"], 2)
            self.assertEqual(len(list((root/"run/videos").rglob("policy.mp4"))), 1)
            resumed = subprocess.run(base + [
                "--resume", str(latest), "--total-steps", "6",
                "--video-interval-evaluations", "0", "--run-dir", str(root/"resume"),
            ], capture_output=True, text=True)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            payload = checkpoint.read(root/"resume/checkpoints/latest.pt")
            self.assertEqual(payload["learning"]["video_interval_evaluations"], 0)
            self.assertEqual(payload["evaluations"], 3)
            self.assertFalse((root/"resume/videos").exists())
