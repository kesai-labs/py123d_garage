"""One H.264 video per named view, streamed to ffmpeg as fragmented MP4 so a killed run stays watchable."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from torch import Tensor

_CRF = 23


class VideoWriter:
    def __init__(self, output_dir: Path, frame_interval: int, fps: float) -> None:
        assert shutil.which("ffmpeg") is not None, (
            "Recording views needs ffmpeg on PATH: install it with "
            "`micromamba install -c conda-forge ffmpeg`, or disable recording."
        )
        self._output_dir = output_dir
        self._frame_interval = frame_interval
        self._fps = fps
        self._encoders: dict[str, subprocess.Popen[bytes]] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

    def is_record_tick(self, step: int) -> bool:
        return step % self._frame_interval == 0

    def write(self, images: dict[str, Tensor]) -> None:
        for name, image in images.items():
            frame = cast(
                "npt.NDArray[np.uint8]",
                image.cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
            ).astype(np.uint8)
            if name not in self._encoders:
                height, width = frame.shape[:2]
                self._encoders[name] = self._start_encoder(name, width, height)
            stdin = self._encoders[name].stdin
            assert stdin is not None
            stdin.write(np.ascontiguousarray(frame).tobytes())

    def _start_encoder(
        self,
        name: str,
        width: int,
        height: int,
    ) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(self._fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-tune",
                "zerolatency",
                "-g",
                "10",
                "-crf",
                str(_CRF),
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                "-movflags",
                "+frag_keyframe+empty_moov+default_base_moof",
                "-an",
                str(self._output_dir / f"{name}.mp4"),
            ],
            stdin=subprocess.PIPE,
        )

    def close(self) -> None:
        for encoder in self._encoders.values():
            assert encoder.stdin is not None
            encoder.stdin.close()
            encoder.wait()
        self._encoders.clear()
