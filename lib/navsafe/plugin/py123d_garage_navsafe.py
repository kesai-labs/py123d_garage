# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The NavSafe policy adapter that plans with a py123d_garage policy, run in its own environment by its agent process."""

from __future__ import annotations

import json
import os
import subprocess
from typing import IO, Any, cast, override

import numpy as np
import numpy.typing as npt
from navsafe.policy.sensor.utils.frames import crop_to_navsim_aspect, renderer_bgr_to_rgb
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS


class GarageAdapter(SensorPolicy):
    """Sends the front camera and the ego state of every replan to the agent process and reads back its plan."""

    @override
    def load_model(self) -> None:
        """Starts the agent process under the py123d_garage environment's python and waits until it is ready."""
        # The simulator's PYTHONPATH and preloads would shadow the agent environment's own packages.
        environment = {
            key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD")
        }
        environment["PYTHONPATH"] = os.environ["PY123D_GARAGE_SRC"]
        self._process = subprocess.Popen(
            [os.environ["PY123D_GARAGE_PYTHON"], "-m", "py123d_garage.evaluation.navsafe.agent", self.checkpoint_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=environment,
        )
        self._agent_stdin = cast(IO[bytes], self._process.stdin)
        self._agent_stdout = cast(IO[bytes], self._process.stdout)
        if self._agent_stdout.readline() != b"READY\n":
            raise RuntimeError("the py123d_garage agent exited before it was ready")
        self.model = self._process

    @override
    def get_camera_configs(self) -> dict[str, dict[str, float]]:
        return {"CAM_F0": NAVSIM_CAM_CONFIGS["CAM_F0"]}

    @override
    def prepare_input(
        self,
        images: dict[str, npt.NDArray[np.uint8]],
        ego_state: dict[str, Any],
        scenario_data: dict[str, Any],
        frame_id: int,
    ) -> tuple[dict[str, Any], npt.NDArray[np.uint8]]:
        """
        Packs the front camera, cut to nuPlan's 1920 x 1080 frame, and the ego's box centre state.

        Args:
            images: the rendered BGR frames by camera.
            ego_state: the evaluator's ego state, in the world frame.
            scenario_data: the scenario, unused.
            frame_id: the simulation frame, 10 per second.

        Returns:
            the request header and the RGB front camera frame.
        """
        image = renderer_bgr_to_rgb(crop_to_navsim_aspect(images["CAM_F0"]))
        request = {
            "frame": int(frame_id),
            "position_xy": [float(value) for value in ego_state["position"][:2]],
            "heading": float(ego_state["heading"]),
            "velocity_xy": [float(value) for value in ego_state["velocity"][:2]],
            "acceleration_xy": [float(value) for value in ego_state["acceleration"][:2]],
            "yaw_rate": float(ego_state["angular_velocity"][2]),
            "image_shape": list(image.shape),
        }
        return request, image

    @override
    def run_inference(self, model_input: tuple[dict[str, Any], npt.NDArray[np.uint8]]) -> dict[str, Any]:
        request, image = model_input
        self._agent_stdin.write(json.dumps(request).encode() + b"\n")
        self._agent_stdin.write(image.tobytes())
        self._agent_stdin.flush()
        reply = self._agent_stdout.readline()
        if not reply:
            raise RuntimeError("the py123d_garage agent exited")
        return json.loads(reply)

    @override
    def parse_output(
        self, model_output: dict[str, Any], ego_state: dict[str, Any]
    ) -> dict[str, npt.NDArray[np.float64]]:
        return {"trajectory": np.asarray(model_output["trajectory"], dtype=np.float64)}
