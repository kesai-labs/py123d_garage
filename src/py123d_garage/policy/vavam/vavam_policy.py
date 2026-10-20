# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Evaluation-only policy that runs the released VaVAM video-action model."""

from __future__ import annotations

import dataclasses
import io
import tempfile
import zipfile
from typing import cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from py123d.api import SceneAPI
from py123d.datatypes import BaseCameraMetadata, CameraID, FThetaCameraMetadata
from py123d.geometry.geometry_index import BoundingBoxSE2Index
from py123d.geometry.transform import abs_to_rel_se2_array
from typing_extensions import override

from py123d_garage.api.abstract_policy import AbstractPolicy, CacheTensorSpec
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    AbstractPredictions,
    NavigationConditioning,
)
from py123d_garage.common.sensor.camera_rectification import FThetaRectifier
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.vavam_config import VavamConfig
from py123d_garage.datatypes.trajectory import derive_yaw_from_positions
from py123d_garage.policy.vavam.network.trajectory_optimizer import (
    TrajectoryOptimizer,
    VehicleConstraints,
    add_heading_to_trajectory,  # pyright: ignore[reportUnknownVariableType]
)
from py123d_garage.policy.vavam.network.transforms import NeuroNCAPTransform
from py123d_garage.policy.vavam.network.video_action_model import (
    VideoActionModelInference,
    load_inference_VAM,
)
from py123d_garage.policy.vavam.visualization import render_view
from py123d_garage.py123d_help.scene_readers import camera_at_anchor, sample_ego_xy, sample_past_camera

# VaVAM command indices: right, left, straight.
_COMMAND_RIGHT = 0
_COMMAND_LEFT = 1
_COMMAND_STRAIGHT = 2

_EVALUATION_ONLY = "VavamPolicy runs the released VaVAM checkpoint for evaluation only."

# Member names inside the VaVAM zip bundle.
_BUNDLE_VAM = "model.pt"
_BUNDLE_TOKENIZER = "tokenizer.jit"


@dataclasses.dataclass(frozen=True)
class VavamFeatures(AbstractFeatures):
    """Context camera frames VaVAM reads."""

    # Context frames normalized to [-1, 1], oldest first, newest last.
    frames: jt.Float[torch.Tensor, "*batch time 3 height width"]


@dataclasses.dataclass(frozen=True)
class VavamLabels(AbstractLabels):
    """Logged future of the ego, used only for drawing."""

    # The logged future positions on the policy's trajectory grid, relative to the anchor.
    trajectory_xy: jt.Float[torch.Tensor, "*batch poses 2"]


@dataclasses.dataclass(frozen=True)
class VavamPredictions(AbstractPredictions):
    """Waypoints VaVAM predicts."""

    # The predicted future positions on the policy's trajectory grid, relative to the anchor.
    waypoints_xy: jt.Float[torch.Tensor, "*batch poses 2"]

    @property
    @override
    def ego_trajectory_xy(self) -> jt.Float[torch.Tensor, "*batch poses 2"]:
        """Inherited, see superclass."""
        return self.waypoints_xy

    @property
    @override
    def ego_trajectory_se2(self) -> jt.Float[torch.Tensor, "*batch poses 3"]:
        """Inherited, see superclass."""
        yaw = derive_yaw_from_positions(self.waypoints_xy)
        return torch.cat([self.waypoints_xy, yaw.unsqueeze(-1)], dim=-1)


class VavamPolicy(AbstractPolicy[VavamFeatures, VavamLabels, VavamPredictions]):
    """Runs the released VaVAM checkpoint for evaluation; training is not supported."""

    def __init__(self, config: PolicyConfig):
        """
        Initializes the VaVAM policy from its config block.

        Args:
            config: policy config holding the vavam_config block.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        assert config.vavam_config is not None, "the target names VavamPolicy but no vavam_config block is declared"
        self._config: VavamConfig = config.vavam_config
        self._vavam_checkpoint_file: str | None = None
        self._image_transform = NeuroNCAPTransform()
        # Keyed by calibration content: live evaluation announces per-scene intrinsics.
        self._rectifiers: dict[tuple[CameraID, int, int, bytes | None], FThetaRectifier] = {}

        optimizer_config = self._config.trajectory_optimizer
        self._trajectory_optimizer: TrajectoryOptimizer | None = None
        self._vehicle_constraints: VehicleConstraints | None = None
        if optimizer_config.enabled:
            self._trajectory_optimizer = TrajectoryOptimizer(
                smoothness_weight=optimizer_config.smoothness_weight,
                deviation_weight=optimizer_config.deviation_weight,
                comfort_weight=optimizer_config.comfort_weight,
                max_iterations=optimizer_config.max_iterations,
                enable_frenet_retiming=optimizer_config.retime_in_frenet,
                retime_alpha=optimizer_config.retime_alpha,
            )
            self._vehicle_constraints = VehicleConstraints(
                max_deviation=optimizer_config.max_deviation,
                max_heading_change=optimizer_config.max_heading_change,
                max_speed=optimizer_config.max_speed,
                max_accel=optimizer_config.max_accel,
                max_abs_yaw_rate=optimizer_config.max_abs_yaw_rate,
                max_abs_yaw_acc=optimizer_config.max_abs_yaw_acc,
                max_lon_acc_pos=optimizer_config.max_lon_acc_pos,
                max_lon_acc_neg=optimizer_config.max_lon_acc_neg,
                max_abs_lon_jerk=optimizer_config.max_abs_lon_jerk,
            )

    @property
    @override
    def policy_config(self) -> VavamConfig:
        """Inherited, see superclass."""
        return self._config

    @override
    def initialize(self, evaluation_checkpoint_file: str | None = None) -> None:
        """
        Loads the VaVAM bundle, which carries its own architecture, instead of the base strict state-dict load.

        Args:
            evaluation_checkpoint_file: path to the VaVAM zip bundle; required.
        """
        if evaluation_checkpoint_file is None:
            raise ValueError(
                "This is inference-only VavamPolicy; requires policy_config.evaluation_checkpoint_files.",
            )
        self._vavam_checkpoint_file = evaluation_checkpoint_file
        self.build_model(from_local_checkpoint=True)

    @override
    def build_model(self, from_local_checkpoint: bool) -> None:
        """
        Loads the video-action model and the tokenizer bytes from the bundle set by initialize.

        Args:
            from_local_checkpoint: unused; the bundle path always comes from initialize.
        """
        del from_local_checkpoint
        assert self._vavam_checkpoint_file is not None, "initialize routes the VaVAM bundle here"
        with zipfile.ZipFile(self._vavam_checkpoint_file) as bundle:
            if not {_BUNDLE_VAM, _BUNDLE_TOKENIZER} <= set(bundle.namelist()):
                raise ValueError(
                    f"{self._vavam_checkpoint_file} is not a VaVAM bundle with "
                    f"{_BUNDLE_VAM} and {_BUNDLE_TOKENIZER}; build it with scripts/pretrained/vavam.sh.",
                )
            with tempfile.TemporaryDirectory() as staging:
                self.vam: VideoActionModelInference = load_inference_VAM(
                    bundle.extract(_BUNDLE_VAM, staging),
                    device="cpu",
                )
            self._tokenizer_bytes = bundle.read(_BUNDLE_TOKENIZER)
        self._tokenizer: torch.jit.ScriptModule | None = None
        self._tokenizer_device: torch.device | None = None

    def _tokenizer_on(self, device: torch.device) -> torch.jit.ScriptModule:
        """
        Returns the tokenizer, jit-loaded onto the device because Module.to() never moves a frozen jit's constants.

        Args:
            device: device the tokenizer must run on.

        Returns:
            the frozen jit tokenizer on that device.
        """
        if self._tokenizer is None or self._tokenizer_device != device:
            self._tokenizer = cast(
                "torch.jit.ScriptModule",
                torch.jit.load(  # pyright: ignore[reportUnknownMemberType]
                    io.BytesIO(self._tokenizer_bytes),
                    map_location=device,
                ),
            )
            self._tokenizer.eval()
            self._tokenizer_device = device
        return self._tokenizer

    def _rectified(
        self,
        metadata: BaseCameraMetadata,
        image: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    ) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
        """
        Rectifies an f-theta camera image to a pinhole view; other cameras pass through.

        Args:
            metadata: metadata of the camera that took the image.
            image: RGB camera image.

        Returns:
            the rectified image at the same resolution.
        """
        if not isinstance(metadata, FThetaCameraMetadata):
            return image
        key = (
            metadata.camera_id,
            metadata.width,
            metadata.height,
            metadata.intrinsics.array.tobytes() if metadata.intrinsics is not None else None,
        )
        rectifier = self._rectifiers.get(key)
        if rectifier is None:
            rectification = self._config.rectification
            rectifier = FThetaRectifier(
                metadata=metadata,
                focal_length_px=rectification.focal_length_px,
                principal_point_px=rectification.principal_point_px,
                radial=rectification.radial,
                tangential=rectification.tangential,
                max_overscan_scale=rectification.max_overscan_scale,
                safety_margin_px=rectification.safety_margin_px,
            )
            self._rectifiers[key] = rectifier
        return rectifier.rectify(image)

    @override
    def build_features(
        self,
        scene_api: SceneAPI,
        cached: dict[str, torch.Tensor],
    ) -> VavamFeatures:
        """Stacks the rectified past context frames and the anchor frame, oldest first."""
        del cached
        config = self._config
        camera_id = config.input_camera[scene_api.scene_metadata.dataset]
        interval_us = config.required_past_camera_interval_us
        anchor = camera_at_anchor(scene_api, camera_id)

        past_cameras = reversed(
            sample_past_camera(
                scene_api=scene_api,
                camera_id=camera_id,
                anchor_timestamp_us=anchor.timestamp.time_us,
                horizon_us=config.required_history_duration_us,
                interval_us=interval_us,
            ),
        )
        images: list[npt.NDArray[np.uint8] | None] = [
            None if camera is None else self._rectified(camera.metadata, camera.image) for camera in past_cameras
        ]
        images.append(self._rectified(anchor.metadata, anchor.image))

        first_served = next(index for index, image in enumerate(images) if image is not None)
        # A rollout younger than the context repeats its oldest served frame.
        previous = images[first_served]
        frames: list[torch.Tensor] = []
        for image in images:
            previous = image if image is not None else previous
            frames.append(self._image_transform(previous))
        return VavamFeatures(frames=torch.stack(frames))

    @override
    def build_labels(
        self,
        scene_api: SceneAPI,
        cached: dict[str, torch.Tensor],
    ) -> VavamLabels:
        """Samples the logged future on the waypoint grid, drawn as ground truth and never trained on."""
        del cached
        trajectory = sample_ego_xy(
            scene_api=scene_api,
            offsets_us=self._config.trajectory_offsets_us,
            relative_to_anchor=True,
        )
        return VavamLabels(trajectory_xy=torch.tensor(trajectory.position_xy_array, dtype=torch.float32))

    @override
    def _permanent_cache_specs(self) -> dict[str, CacheTensorSpec]:
        """Inherited, see superclass; VaVAM caches nothing."""
        return {}

    @override
    def forward(
        self,
        features: VavamFeatures,
        navigation: NavigationConditioning,
    ) -> VavamPredictions:
        """
        Tokenizes the context frames and denoises one trajectory per sample.

        Args:
            features: batched context frames.
            navigation: navigation conditioning; the lateral offset of the first target point picks the command.

        Returns:
            the predicted waypoints, optionally smoothed by the trajectory optimizer.
        """
        config = self._config
        batch_size, context_length = features.frames.shape[0], features.frames.shape[1]
        device = features.frames.device
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        lateral_m = navigation.target_points[:, 0, 1]
        command = torch.full((batch_size, 1), _COMMAND_STRAIGHT, dtype=torch.long, device=device)
        command[lateral_m > config.command_lateral_threshold_m] = _COMMAND_LEFT
        command[lateral_m < -config.command_lateral_threshold_m] = _COMMAND_RIGHT

        # Autocast cannot cast the frozen jit tokenizer's baked constants, so it runs fp32.
        tokens = self._tokenizer_on(device)(features.frames.flatten(0, 1))
        tokens = tokens.view(batch_size, context_length, *tokens.shape[1:])
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            waypoints = self.vam.forward_inference(tokens, cast("torch.LongTensor", command), dtype)
        waypoints_xy = waypoints.squeeze(1).float()
        if self._trajectory_optimizer is not None:
            waypoints_xy = self._optimized_waypoints(waypoints_xy)
        return VavamPredictions(waypoints_xy=waypoints_xy)

    def _optimized_waypoints(
        self,
        waypoints_xy: jt.Float[torch.Tensor, "*batch poses 2"],
    ) -> jt.Float[torch.Tensor, "*batch poses 2"]:
        """
        Smooths each sample's waypoints with the trajectory optimizer, keeping the raw ones where it fails.

        Args:
            waypoints_xy: predicted waypoints.

        Returns:
            the optimized waypoints with the input's dtype and device.
        """
        assert self._trajectory_optimizer is not None
        assert self._vehicle_constraints is not None
        optimized = cast(
            "npt.NDArray[np.float32]",
            waypoints_xy.detach().cpu().numpy().copy(),  # pyright: ignore[reportUnknownMemberType]
        )
        for index in range(optimized.shape[0]):
            result = self._trajectory_optimizer.optimize(  # pyright: ignore[reportUnknownMemberType]
                trajectory=add_heading_to_trajectory(optimized[index]),
                time_step=self._config.trajectory_offsets_us[0] / 1_000_000,
                vehicle_constraints=self._vehicle_constraints,
            )
            if result.success:
                trajectory = cast("npt.NDArray[np.float64]", result.trajectory)  # pyright: ignore[reportUnknownMemberType]
                optimized[index] = trajectory[:, :2]
        return torch.as_tensor(optimized, dtype=waypoints_xy.dtype, device=waypoints_xy.device)

    @override
    def compute_loss(
        self,
        labels: VavamLabels,
        predictions: VavamPredictions,
    ) -> dict[str, torch.Tensor]:
        """Not supported; VaVAM is evaluation-only."""
        raise NotImplementedError(_EVALUATION_ONLY)

    @override
    def loss_weights(self, epoch: int) -> dict[str, float]:
        """Not supported; VaVAM is evaluation-only."""
        raise NotImplementedError(_EVALUATION_ONLY)

    @override
    def visualize_batch(
        self,
        features: VavamFeatures,
        labels: VavamLabels | None,
        navigation: NavigationConditioning,
        predictions: VavamPredictions,
        scene_api: SceneAPI,
    ) -> dict[str, torch.Tensor]:
        """Renders the first sample's newest context frame over a BEV sheet with its trajectories."""
        ego_state = scene_api.get_ego_state_se3_at_iteration(0)
        assert ego_state is not None, "Ego state should be available for visualization!"
        ego_box = ego_state.bounding_box_se2.array.copy()
        ego_box[BoundingBoxSE2Index.SE2] = abs_to_rel_se2_array(
            origin=ego_state.rear_axle_se2,
            pose_se2_array=ego_box[None, BoundingBoxSE2Index.SE2],
        )[0]
        camera_rgb = cast(
            "npt.NDArray[np.uint8]",
            ((features.frames[0, -1].detach().cpu() + 1) * 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy(),  # pyright: ignore[reportUnknownMemberType]
        )
        image = render_view(
            camera_rgb,
            _first_sample(predictions.waypoints_xy),
            _first_sample(labels.trajectory_xy) if labels is not None else None,
            _first_sample(navigation.target_points),
            ego_box.astype(np.float32),
            self._config.visualization_config,
        )
        return {"predictions": torch.from_numpy(image)}  # pyright: ignore[reportUnknownMemberType]


def _first_sample(
    tensor: jt.Float[torch.Tensor, " batch *rest"],
) -> jt.Float32[npt.NDArray[np.float32], "*rest"]:
    """
    Converts the first sample of a batched tensor to a float32 numpy array.

    Args:
        tensor: batched tensor.

    Returns:
        the first sample as a float32 array.
    """
    return cast(
        "npt.NDArray[np.float32]",
        tensor[0].detach().cpu().numpy().astype(np.float32),  # pyright: ignore[reportUnknownMemberType]
    )
