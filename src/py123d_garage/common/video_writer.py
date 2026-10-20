# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""One H.264 video per named view, streamed to ffmpeg as fragmented MP4 so a killed run stays watchable."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from torch import Tensor

_CRF = 23


class VideoWriter:
    """Records every n-th tick of each named view into its own MP4."""

    def __init__(self, output_dir: Path, frame_interval: int, fps: float) -> None:
        """
        Initializes the writer; the encoders start with the first frame of each view.

        Args:
            output_dir: directory the <view name>.mp4 files are written to.
            frame_interval: number of ticks between recorded frames.
            fps: frame rate of the videos.
        """
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
        """
        Checks whether a tick gets recorded.

        Args:
            step: the tick index.

        Returns:
            True on every frame_interval-th tick.
        """
        return step % self._frame_interval == 0

    def write(self, images: dict[str, jt.Shaped[Tensor, "height width 3"]]) -> None:
        """
        Appends one RGB frame to each named view's video.

        Args:
            images: uint8 RGB frames by view name.
        """
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
        """
        Starts the ffmpeg process of one view.

        Args:
            name: the view name, which names the file.
            width: frame width in pixels.
            height: frame height in pixels.

        Returns:
            the ffmpeg process, reading raw RGB frames on stdin.
        """
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
                "-threads",
                "2",
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

    def close(self, prefix: str = "") -> None:
        """
        Finishes the videos whose name starts with prefix.

        Args:
            prefix: name prefix of the videos to finish; empty finishes all of them.
        """
        for name in [name for name in self._encoders if name.startswith(prefix)]:
            encoder = self._encoders.pop(name)
            assert encoder.stdin is not None
            encoder.stdin.close()
            encoder.wait()
