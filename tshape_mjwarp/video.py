"""Stream offscreen MuJoCo frames into an atomic H.264 MP4 via FFmpeg."""

from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np


class EpisodeVideo:
    def __init__(self, physics, path, width=1920, height=1080, crf=18):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 or v % 2
               for v in (width, height)):
            raise ValueError("video dimensions must be even integers >= 2")
        if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
            raise ValueError("video CRF must be an integer in [0, 51]")
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("Video recording needs FFmpeg on PATH (Ubuntu: sudo apt install ffmpeg)")
        self.physics, self.path = physics, Path(path)
        self.width, self.height, self.crf = width, height, crf
        self.fps = 1.0 / physics.control_dt
        self.frames = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.path.with_suffix(".partial.mp4")
        self.renderer = self.process = self.errors = None
        try:
            model = physics.model
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            self.renderer = mujoco.Renderer(model, height=height, width=width)
            self.errors = tempfile.TemporaryFile()
            self.process = subprocess.Popen([
                executable, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                "-framerate", f"{self.fps:.12g}", "-i", "pipe:0", "-an",
                "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2",
                str(self.temporary_path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors)
        except Exception:
            self.close(abort=True)
            raise

    def capture(self):
        self.renderer.update_scene(self.physics.data, camera="top_view")
        frame = self.renderer.render()
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self.close(abort=True)
            raise RuntimeError("FFmpeg stopped while encoding the evaluation video") from None
        self.frames += 1

    def metadata(self):
        return {"path": str(self.path), "width": self.width, "height": self.height,
                "fps": self.fps, "frames": self.frames, "codec": "h264", "crf": self.crf}

    def close(self, abort=False):
        error = None
        try:
            if self.process is not None:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                if abort:
                    self.process.terminate()
                status = self.process.wait()
                self.errors.seek(0)
                message = self.errors.read().decode("utf-8", errors="replace").strip()
                self.process = None
                if not abort and (status != 0 or not self.frames):
                    error = RuntimeError(f"FFmpeg could not finish video: {message or 'no frames'}")
                elif not abort:
                    self.temporary_path.replace(self.path)
        finally:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            if self.errors is not None:
                self.errors.close()
                self.errors = None
            if abort or error is not None:
                self.temporary_path.unlink(missing_ok=True)
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(abort=exc_type is not None)


