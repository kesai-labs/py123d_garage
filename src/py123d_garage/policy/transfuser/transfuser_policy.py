# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Protocol, cast, runtime_checkable

import jaxtyping as jt
import torch
import torch.nn.functional as F
from py123d.api import SceneAPI
from py123d.geometry import PoseSE2Index
from typing_extensions import override

from py123d_garage.api.abstract_offline_data_source_config import HAS_BOX_ANNOTATIONS
from py123d_garage.api.abstract_policy import (
    AbstractPolicy,
    CacheTensorSpec,
    CameraAugmentation,
    normalize_loss_weights,
)
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.cache.codec import (
    JpegCodec,
    PngCodec,
    ZlibCodec,
)
from py123d_garage.common.config_help.import_string import import_string
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.features import (
    TransfuserFeatures,
    build_acceleration_vector,
    build_camera_feature,
    build_ego_history,
    build_lidar_feature,
    build_velocity,
    build_velocity_vector,
)
from py123d_garage.policy.transfuser.labels import (
    TransfuserLabels,
    build_bev_labels,
    build_path_label,
    build_perspective_labels,
    build_trajectory_label,
    depth_codes_to_m,
    depth_m_to_codes,
    merge_semantic_classes,
)
from py123d_garage.policy.transfuser.network.decoder.bev_semantic_decoder import (
    BEVSemanticDecoder,
    BevSemanticLabels,
)
from py123d_garage.policy.transfuser.network.decoder.center_net_decoder import (
    CenterNetBoundingBoxPrediction,
    CenterNetDecoder,
    CenterNetLabels,
)
from py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder import (
    ROLLOUT_COLUMNS,
    WAYPOINT_COLUMNS,
)
from py123d_garage.policy.transfuser.network.decoder.perspective_decoder import (
    PerspectiveDecoder,
)
from py123d_garage.policy.transfuser.network.decoder.regression_planning_decoder import (
    RegressionPlanningDecoder,
)
from py123d_garage.policy.transfuser.network.decoder.unicycle import fit_actions
from py123d_garage.policy.transfuser.network.encoder.streaming_transfuser import (
    FrameTokens,
    StreamingTransfuserBackbone,
)
from py123d_garage.policy.transfuser.network.encoder.transfuser import (
    TransfuserBackbone,
)
from py123d_garage.policy.transfuser.predictions import TransfuserPredictions
from py123d_garage.policy.transfuser.visualization import (
    bev_semantic_classes_of,
    ego_box_of,
    ego_frame_route_of,
    ground_truth_boxes_of,
    render_bev_maps,
    render_ground_truth,
    render_prediction,
)

LOG = logging.getLogger(__name__)


@runtime_checkable
class _TaskHead(Protocol):
    """A decoder that scores its own prediction against its label."""

    def compute_loss(self, predictions: Any, label: Any, /) -> dict[str, torch.Tensor]:
        """
        Computes the head's losses.

        Args:
            predictions: the head's prediction.
            label: the head's label.

        Returns:
            scalar losses by name.
        """
        ...

    def compute_metrics(self, predictions: Any, label: Any, /) -> dict[str, torch.Tensor]:
        """
        Computes the head's quality metrics.

        Args:
            predictions: the head's prediction.
            label: the head's label.

        Returns:
            scalar metrics by name.
        """
        ...


def _cached_depth_codes(
    depth: jt.Shaped[torch.Tensor, "*batch height width"],
    config: TransfuserConfig,
) -> jt.UInt8[torch.Tensor, "*batch height width"]:
    """
    Cached depth as 8-bit codes, converting meters from older stores.

    Args:
        depth: the cached depth label, codes or meters.
        config: TransFuser config.

    Returns:
        the depth as 8-bit linear codes.
    """
    if depth.dtype == torch.uint8:
        return depth
    return depth_m_to_codes(depth, config)


class TransfuserPolicy(
    AbstractPolicy[TransfuserFeatures, TransfuserLabels, TransfuserPredictions],
):
    """TransFuser: a camera and optional LiDAR backbone with a planning decoder and auxiliary perception heads."""

    def __init__(self, config: PolicyConfig):
        """
        Stores the config; build_model creates the modules.

        Args:
            config: policy config holding a transfuser_config block.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        assert config.transfuser_config is not None, (
            "the target names TransfuserPolicy but no transfuser_config block is declared"
        )
        self._config: TransfuserConfig = config.transfuser_config

    @property
    @override
    def policy_config(self) -> TransfuserConfig:
        """Inherited, see superclass."""
        return self._config

    @override
    def build_model(self, from_local_checkpoint: bool) -> None:
        """Builds the backbone and every config-enabled head."""
        config = self._config
        backbone_class: type[TransfuserBackbone] = import_string(
            config.backbone_config.backbone_target,
        )
        self.backbone = backbone_class(
            config,
            image_encoder_pretrained=config.backbone_config.image_encoder_pretrained and not from_local_checkpoint,
        )
        if config.backbone_config.freeze_backbone:
            self.backbone.requires_grad_(False)

        if config.perspective_config.use_semantic:
            self.semantic_decoder = PerspectiveDecoder(
                config=config,
                in_channels=self.backbone.num_image_features,
                out_channels=config.perspective_config.num_semantic_classes,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="semantic",
            )

        if config.perspective_config.use_depth:
            self.depth_decoder = PerspectiveDecoder(
                config=config,
                in_channels=self.backbone.num_image_features,
                out_channels=1,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="depth",
            )

        if config.bev_semantic_config.use_bev_semantic:
            self.bev_semantic_decoder = BEVSemanticDecoder(config)

        if config.box_detection_config.detect_boxes:
            self.center_net_decoder = CenterNetDecoder(config)

        if config.planning_config.use_planning_decoder:
            self.planning_decoder = self._new_planning_decoder()
        conditioning = config.planning_conditioning_config
        if conditioning.unconditional_epochs or conditioning.target_point_dropout_probabilities is not None:
            # The epoch's dropout per target point; a tensor so the recorded step sees each epoch's value.
            self.register_buffer(
                "target_point_dropout",
                torch.ones(len(config.required_target_point_distances_m))
                if conditioning.unconditional_epochs
                else torch.tensor(conditioning.target_point_dropout_probabilities),
                persistent=False,
            )

    @override
    def prepare_training(self) -> None:
        """Swaps the kernel_patches custom kernels into the backbone and the decoders; evaluation runs without them."""
        for kernel_patch in self._config.backbone_config.kernel_patches:
            import_string(kernel_patch)(self.backbone)
        if self._config.planning_config.use_planning_decoder:
            for kernel_patch in self._config.planning_config.kernel_patches:
                import_string(kernel_patch)(self.planning_decoder)
        if self._config.bev_semantic_config.use_bev_semantic:
            for kernel_patch in self._config.bev_semantic_config.kernel_patches:
                import_string(kernel_patch)(self.bev_semantic_decoder)
        if self._config.box_detection_config.detect_boxes:
            for kernel_patch in self._config.box_detection_config.kernel_patches:
                import_string(kernel_patch)(self.center_net_decoder)

    def _new_planning_decoder(self) -> RegressionPlanningDecoder:
        """
        Builds the planning decoder named by planning_decoder_target on the backbone's channels.

        Returns:
            the planning decoder.
        """
        config = self._config
        planning_decoder_class: type[RegressionPlanningDecoder] = import_string(
            config.planning_config.planning_decoder_target,
        )
        return planning_decoder_class(
            input_bev_channels=self.backbone.num_lidar_features,
            config=config,
            input_register_channels=self.backbone.num_register_features,
        )

    @override
    def on_train_epoch_start(self, epoch: int) -> list[torch.nn.Parameter]:
        """Sets the epoch's target point dropout."""
        conditioning = self._config.planning_conditioning_config
        if conditioning.unconditional_epochs or conditioning.target_point_dropout_probabilities is not None:
            num_target_points = len(self._config.required_target_point_distances_m)
            if epoch < conditioning.unconditional_epochs:
                dropout = [1.0] * num_target_points
            elif conditioning.target_point_dropout_probabilities is not None:
                dropout = conditioning.target_point_dropout_probabilities
            else:
                dropout = [conditioning.target_point_dropout_probability] * num_target_points
            cast(torch.Tensor, self.target_point_dropout).copy_(torch.tensor(dropout))
            LOG.info(f"Target point dropout {dropout} in epoch {epoch}.")
        return []

    @override
    def train(self, mode: bool = True) -> TransfuserPolicy:
        """Keeps a frozen backbone in eval mode so its batch norms and dropouts stay fixed."""
        super().train(mode)
        if self._config.backbone_config.freeze_backbone and "backbone" in self._modules:
            self.backbone.eval()
        return self

    @override
    def cache_signature(self, dataset: str) -> dict[str, dict[str, str]]:
        """Narrows the camera entries to the dataset's cameras."""
        cameras = self._config.camera_config.input_cameras[dataset]
        input_cameras = str([camera_id.name for camera_id in cameras])
        signature = {tensor_name: dict(fields) for tensor_name, fields in super().cache_signature(dataset).items()}
        for fields in signature.values():
            if "input_cameras" in fields:
                fields["input_cameras"] = input_cameras
        return signature

    @override
    def build_features(
        self,
        scene_api: SceneAPI,
        cached: dict[str, torch.Tensor],
    ) -> TransfuserFeatures:
        """Inherited, see superclass."""
        config = self._config
        lidar_feature: torch.Tensor | None = None
        if not config.backbone_config.latent:
            lidar_feature = (
                cached["lidar_feature"] if "lidar_feature" in cached else build_lidar_feature(scene_api, config)
            )
        return TransfuserFeatures(
            # Older stores hold a non-uint8 dtype.
            camera_feature=cached["camera_feature"].to(torch.uint8)
            if "camera_feature" in cached
            else build_camera_feature(scene_api, config),
            velocity=cached["velocity"] if "velocity" in cached else build_velocity(scene_api, config),
            velocity_vector=cached["velocity_vector"]
            if "velocity_vector" in cached
            else build_velocity_vector(scene_api, config),
            acceleration_vector=cached["acceleration_vector"]
            if "acceleration_vector" in cached
            else build_acceleration_vector(scene_api, config),
            ego_history=(cached["ego_history"] if "ego_history" in cached else build_ego_history(scene_api, config))
            if config.ego_history_num_steps
            else None,
            lidar_feature=lidar_feature,
        )

    @override
    def build_labels(
        self,
        scene_api: SceneAPI,
        cached: dict[str, torch.Tensor],
    ) -> TransfuserLabels:
        """Inherited, see superclass."""
        config = self._config
        trajectory: torch.Tensor | None = None
        action: torch.Tensor | None = None
        path: torch.Tensor | None = None
        if config.planning_config.use_planning_decoder:
            trajectory = cached["trajectory"] if "trajectory" in cached else build_trajectory_label(scene_api, config)
            path_config = config.planning_config.path_config
            if path_config is not None:
                path = cached["path"] if "path" in cached else build_path_label(scene_api, path_config)
            action_config = config.planning_config.unicycle_action_config
            if action_config is not None:
                if "action" in cached:
                    action = cached["action"]
                else:
                    velocity = cached["velocity"] if "velocity" in cached else build_velocity(scene_api, config)
                    action = fit_actions(trajectory[: config.trajectory_num_steps], velocity[0], action_config)
        # Set by the training dataset from its source config; True when absent.
        has_box_annotations = cached.get(HAS_BOX_ANNOTATIONS, torch.tensor(True))
        use_bev_semantic = config.bev_semantic_config.use_bev_semantic
        detect_boxes = config.box_detection_config.detect_boxes
        bev_semantic: BevSemanticLabels | None = None
        center_net: CenterNetLabels | None = None
        # The store holds both BEV label groups or neither; the session cache holds neither.
        bev_stored = ("bev_semantic" in cached) if use_bev_semantic else ("center_net_heatmap" in cached)
        if (use_bev_semantic or detect_boxes) and not bev_stored:
            bev_semantic, center_net = build_bev_labels(scene_api, config)
        elif use_bev_semantic or detect_boxes:
            if use_bev_semantic:
                bev_semantic = BevSemanticLabels(
                    # Older stores hold a non-uint8 dtype.
                    bev_semantic=cached["bev_semantic"].to(torch.uint8),
                    bev_semantic_has_map=cached["bev_semantic_has_map"],
                    bev_semantic_has_box_annotations=has_box_annotations,
                )
            if detect_boxes:
                center_net = CenterNetLabels(
                    center_net_heatmap=cached["center_net_heatmap"],
                    center_net_wh=cached["center_net_wh"],
                    center_net_offset=cached["center_net_offset"],
                    # Older stores hold a non-uint8 dtype.
                    center_net_yaw_class=cached["center_net_yaw_class"].to(torch.uint8),
                    center_net_yaw_res=cached["center_net_yaw_res"],
                    center_net_pixel_weight=cached["center_net_pixel_weight"],
                    center_net_avg_factor=cached["center_net_avg_factor"],
                    center_net_has_box_annotations=has_box_annotations,
                    center_net_velocity=cached["center_net_velocity"]
                    if config.box_detection_config.predict_box_velocity
                    else None,
                )
        use_semantic = config.perspective_config.use_semantic
        use_depth = config.perspective_config.use_depth
        semantic: torch.Tensor | None = None
        depth: torch.Tensor | None = None
        perspective_stored = ("semantic" in cached) if use_semantic else ("depth" in cached)
        if (use_semantic or use_depth) and not perspective_stored:
            semantic, depth = build_perspective_labels(scene_api, config)
        elif use_semantic or use_depth:
            semantic = cached["semantic"] if use_semantic else None
            depth = _cached_depth_codes(cached["depth"], config) if use_depth else None
        return TransfuserLabels(
            trajectory=trajectory,
            action=action,
            path=path,
            semantic=semantic,
            depth=depth,
            bev_semantic=bev_semantic,
            center_net=center_net,
        )

    @override
    def session_cache_names(self) -> frozenset[str]:
        """Inherited, see superclass."""
        config = self._config
        names = {"velocity", "velocity_vector", "acceleration_vector", "target_points", "target_point_headings"}
        if config.ego_history_num_steps:
            names.add("ego_history")
        if config.planning_config.use_planning_decoder:
            names.add("trajectory")
            if config.planning_config.unicycle_action_config is not None:
                names.add("action")
            if config.planning_config.path_config is not None:
                names.add("path")
        return frozenset(names)

    @override
    def _permanent_cache_specs(self) -> dict[str, CacheTensorSpec]:
        """
        Codec and builder config values of each cached tensor.

        Returns:
            spec by tensor name.
        """
        # NOTE@ln2697: builder code is not part of the signature; byte-changing edits need force_cache_rebuild.
        config = self._config
        lidar = config.lidar_config
        bev = config.bev_semantic_config
        boxes = config.box_detection_config
        camera = config.camera_config
        perspective = config.perspective_config

        bev_grid = {
            "bev_pixels_per_meter": str(lidar.bev_pixels_per_meter),
            "bev_min_x_m": str(lidar.bev_min_x_m),
            "bev_max_x_m": str(lidar.bev_max_x_m),
            "bev_min_y_m": str(lidar.bev_min_y_m),
            "bev_max_y_m": str(lidar.bev_max_y_m),
        }
        center_net = {
            **bev_grid,
            "bev_downsample_factor": str(bev.bev_downsample_factor),
            "detection_classes": str(boxes.detection_classes),
            "num_yaw_bins": str(boxes.num_yaw_bins),
        }
        input_cameras = str(
            {
                dataset: [camera_id.name for camera_id in cameras]
                for dataset, cameras in sorted(camera.input_cameras.items())
            },
        )
        perspective_geometry = {
            "input_cameras": input_cameras,
            "image_width": str(camera.image_width),
            "image_height": str(camera.image_height),
            "perspective_downsample_factor": str(
                perspective.perspective_downsample_factor,
            ),
            "past_camera_rows": str((camera.past_camera_offset_us, camera.num_past_camera_rows)),
        }

        camera_signature = {
            "input_cameras": input_cameras,
            "image_width": str(camera.image_width),
            "image_height": str(camera.image_height),
            "cache_jpeg_quality": str(camera.cache_jpeg_quality),
            "past_camera_rows": str((camera.past_camera_offset_us, camera.num_past_camera_rows)),
        }
        tensors: dict[str, CacheTensorSpec] = {
            # Lossy JPEG because the camera is by far the largest cached tensor.
            "camera_feature": CacheTensorSpec(
                codec=JpegCodec(
                    quality=camera.cache_jpeg_quality,
                    quantization_scale=1,
                ),
                signature=camera_signature,
            ),
        }
        if not config.backbone_config.latent:
            tensors["lidar_feature"] = CacheTensorSpec(
                codec=PngCodec(quantization_scale=255),
                signature={
                    **bev_grid,
                    "max_lidar_points_per_bev_pixel": str(
                        lidar.max_lidar_points_per_bev_pixel,
                    ),
                    "lidar_max_height_m": str(lidar.lidar_max_height_m),
                    "lidar_min_height_m": str(lidar.lidar_min_height_m),
                    "lidar_ground_z_m": str(lidar.lidar_ground_z_m),
                    "remove_lidar_ground_points": str(
                        lidar.remove_lidar_ground_points,
                    ),
                    "ground_removal": str(lidar.ground_removal),
                    "lidar_horizon_us": str(lidar.lidar_horizon_us),
                    "lidar_interval_us": str(lidar.lidar_interval_us),
                },
            )
        if bev.use_bev_semantic:
            tensors["bev_semantic"] = CacheTensorSpec(
                codec=PngCodec(quantization_scale=1),
                signature={
                    **bev_grid,
                    "bev_semantic_classes": str(
                        bev.selected_bev_semantic_classes,
                    ),
                    "pedestrian_bev_extent_scale": str(
                        bev.pedestrian_bev_extent_scale,
                    ),
                    "pedestrian_bev_min_extent_m": str(
                        bev.pedestrian_bev_min_extent_m,
                    ),
                    "map_z_gate_m": str(bev.map_z_gate_m),
                },
            )
            tensors["bev_semantic_has_map"] = CacheTensorSpec(codec=ZlibCodec(), signature={})
        if boxes.detect_boxes:
            for tensor_name in (
                "center_net_heatmap",
                "center_net_wh",
                "center_net_offset",
                "center_net_yaw_class",
                "center_net_yaw_res",
                "center_net_pixel_weight",
                "center_net_avg_factor",
            ):
                tensors[tensor_name] = CacheTensorSpec(
                    codec=ZlibCodec(),
                    signature={**center_net},
                )
            if config.box_detection_config.predict_box_velocity:
                tensors["center_net_velocity"] = CacheTensorSpec(
                    codec=ZlibCodec(),
                    signature={
                        **center_net,
                        # The velocity label depends on the accumulated lidar sweeps.
                        "lidar_horizon_us": str(lidar.lidar_horizon_us),
                        "lidar_interval_us": str(lidar.lidar_interval_us),
                    },
                )
        if perspective.use_semantic:
            tensors["semantic"] = CacheTensorSpec(
                codec=PngCodec(quantization_scale=1),
                signature={
                    **perspective_geometry,
                    **(
                        {"semantic_class_ids": str(perspective.semantic_class_ids)}
                        if perspective.semantic_class_ids is not None
                        else {}
                    ),
                },
            )
        if perspective.use_depth:
            tensors["depth"] = CacheTensorSpec(
                # The label is already 8-bit codes, so the codec stores them as they are.
                codec=PngCodec(quantization_scale=1),
                signature={
                    **perspective_geometry,
                    "depth_max_m": str(perspective.depth_max_m),
                },
            )
        return tensors

    @override
    def augment_features(
        self,
        features: TransfuserFeatures,
        augment_camera: CameraAugmentation,
    ) -> TransfuserFeatures:
        """Augments only the stitched camera image, given its tile layout."""
        camera = self._config.camera_config
        num_cameras = max((len(camera_ids) for camera_ids in camera.input_cameras.values()), default=1)
        return replace(
            features,
            camera_feature=augment_camera(
                features.camera_feature,
                num_cameras,
                1 + camera.num_past_camera_rows if camera.past_camera_offset_us else 1,
                camera.front_camera_index,
            ),
        )

    @override
    def forward(
        self,
        features: TransfuserFeatures,
        navigation: NavigationConditioning,
    ) -> TransfuserPredictions:
        """Runs the backbone and every config-enabled head."""
        lidar = None if self._config.backbone_config.latent else features.lidar_feature
        bev_features, image_features, fusion_registers = self.backbone(features.camera_feature.float(), lidar)
        return self._forward_heads(features, navigation, bev_features, image_features, fusion_registers)

    def forward_streaming(
        self,
        features: TransfuserFeatures,
        navigation: NavigationConditioning,
        past_frame_tokens: list[FrameTokens],
    ) -> tuple[TransfuserPredictions, FrameTokens]:
        """
        Runs the model on the current frame and earlier calls' tokens, so a streaming backbone encodes each frame once.

        Args:
            features: batched feature bundle; camera_feature is the canvas or just its current (bottom) row.
            navigation: batched navigation conditioning.
            past_frame_tokens: the tokens of the frames 1, 2, ... times past_camera_offset_us back, nearest first, as
                earlier calls returned them; fewer at the start of a rollout.

        Returns:
            the prediction bundle and the current frame's tokens, for the later calls.
        """
        backbone = self.backbone
        assert isinstance(backbone, StreamingTransfuserBackbone), "forward_streaming needs a streaming backbone"
        current_frame = features.camera_feature[..., -backbone.row_height :, :].float()
        bev_features, image_features, fusion_registers, frame_tokens = backbone.stream(
            current_frame,
            past_frame_tokens,
            None,
        )
        predictions = self._forward_heads(features, navigation, bev_features, image_features, fusion_registers)
        return predictions, frame_tokens

    def _forward_heads(
        self,
        features: TransfuserFeatures,
        navigation: NavigationConditioning,
        bev_features: torch.Tensor,
        image_features: torch.Tensor,
        fusion_registers: torch.Tensor | None,
    ) -> TransfuserPredictions:
        """
        Runs every config-enabled head on the backbone's outputs.

        Args:
            features: batched feature bundle from build_features.
            navigation: batched navigation conditioning.
            bev_features: the backbone's BEV features.
            image_features: the backbone's image features.
            fusion_registers: the backbone's last-stage registers, None without registers.

        Returns:
            the prediction bundle.
        """
        config = self._config
        conditioning = config.planning_conditioning_config
        velocity = features.velocity if conditioning.use_velocity else None
        speed_relative_noise_std = conditioning.speed_relative_noise_std
        if self.training and velocity is not None and speed_relative_noise_std > 0.0:
            speed_noise_scales = 1.0 + speed_relative_noise_std * torch.randn_like(velocity)
            velocity = (velocity * speed_noise_scales).clamp(min=0.0)
        velocity_vector = features.velocity_vector if conditioning.use_velocity_vector else None
        acceleration_vector = features.acceleration_vector if conditioning.use_acceleration_vector else None

        target_points = navigation.target_points
        target_point_lateral_noise_std_m = conditioning.target_point_lateral_noise_std_m
        if self.training and target_point_lateral_noise_std_m > 0.0:
            target_point_headings = navigation.target_point_headings
            route_left_normals = torch.stack(
                [-torch.sin(target_point_headings), torch.cos(target_point_headings)],
                dim=-1,
            )
            noise_shape = target_point_headings.shape
            if conditioning.target_point_lateral_noise_correlated:
                noise_shape = (*noise_shape[:-1], 1)
            target_point_lateral_offsets_m = target_point_lateral_noise_std_m * torch.randn(
                noise_shape,
                device=target_point_headings.device,
                dtype=target_point_headings.dtype,
            ).expand_as(target_point_headings)
            target_points = target_points + target_point_lateral_offsets_m.unsqueeze(-1) * route_left_normals
        dropout = conditioning.target_point_dropout_probability
        if self.training and (
            conditioning.unconditional_epochs or conditioning.target_point_dropout_probabilities is not None
        ):
            dropped = torch.rand(target_points.shape[:2], device=target_points.device) < cast(
                torch.Tensor,
                self.target_point_dropout,
            )
            target_points = target_points.masked_fill(dropped[..., None], torch.nan)
        elif dropout >= 1.0:
            target_points = torch.full_like(target_points, torch.nan)
        elif self.training and dropout > 0.0:
            dropped = torch.rand(target_points.shape[:2], device=target_points.device) < dropout
            target_points = target_points.masked_fill(dropped[..., None], torch.nan)
        served = conditioning.test_time_target_point_distances_m
        if not self.training and served is not None:
            unserved = torch.tensor(
                [distance_m not in served for distance_m in config.required_target_point_distances_m],
                device=target_points.device,
            )
            target_points = target_points.masked_fill(unserved[None, :, None], torch.nan)

        trajectory: torch.Tensor | None = None
        if config.planning_config.use_planning_decoder:
            trajectory = self.planning_decoder(
                bev_features,
                target_points,
                velocity,
                velocity_vector,
                acceleration_vector,
                features.ego_history,
                fusion_registers=fusion_registers,
            )
        path: torch.Tensor | None = None
        if trajectory is not None and config.planning_config.path_config is not None and not torch.is_grad_enabled():
            # The path decoder outputs the poses followed by the path points.
            trajectory, path = (
                trajectory[:, : config.trajectory_num_steps],
                trajectory[:, config.trajectory_num_steps :, : len(PoseSE2Index)],
            )
        waypoints: torch.Tensor | None = None
        rollout: torch.Tensor | None = None
        if (
            trajectory is not None
            and config.planning_config.unicycle_action_config is not None
            and not torch.is_grad_enabled()
        ):
            rollout = trajectory[..., ROLLOUT_COLUMNS]
            if config.planning_config.predict_waypoints:
                waypoints = trajectory[..., WAYPOINT_COLUMNS]
        semantic: torch.Tensor | None = None
        if config.perspective_config.use_semantic:
            semantic = self.semantic_decoder(image_features)
        depth: torch.Tensor | None = None
        if config.perspective_config.use_depth:
            depth = self.depth_decoder(image_features)

        boxes: CenterNetBoundingBoxPrediction | None = None
        bev_semantic: torch.Tensor | None = None
        if config.box_detection_config.detect_boxes or config.bev_semantic_config.use_bev_semantic:
            bev_feature_grid = self.backbone.top_down(bev_features)
            if config.box_detection_config.detect_boxes:
                boxes = self.center_net_decoder(bev_feature_grid)
            if config.bev_semantic_config.use_bev_semantic:
                bev_semantic = self.bev_semantic_decoder(bev_feature_grid)

        return TransfuserPredictions(
            trajectory=trajectory,
            path=path,
            waypoints=waypoints,
            rollout=rollout,
            semantic=semantic,
            depth=depth,
            bev_semantic=bev_semantic,
            boxes=boxes,
        )

    def _enabled_heads(
        self,
        labels: TransfuserLabels,
        predictions: TransfuserPredictions,
    ) -> list[tuple[_TaskHead, Any, Any]]:
        """
        One (decoder, prediction, label) row per config-enabled head.

        Args:
            labels: batched labels from build_labels.
            predictions: predictions from forward.

        Returns:
            each enabled head's decoder with its prediction and label.
        """
        config = self._config
        rows: list[tuple[_TaskHead, Any, Any]] = []
        if config.perspective_config.use_semantic:
            semantic = None if labels.semantic is None else merge_semantic_classes(labels.semantic, config)
            rows.append((self.semantic_decoder, predictions.semantic, semantic))
        if config.perspective_config.use_depth:
            depth = None if labels.depth is None else depth_codes_to_m(labels.depth, config)
            rows.append((self.depth_decoder, predictions.depth, depth))
        if config.bev_semantic_config.use_bev_semantic:
            rows.append((self.bev_semantic_decoder, predictions.bev_semantic, labels.bev_semantic))
        if config.box_detection_config.detect_boxes:
            rows.append((self.center_net_decoder, predictions.boxes, labels.center_net))
        if config.planning_config.use_planning_decoder:
            trajectory_label = labels.trajectory
            if trajectory_label is not None and labels.action is not None:
                trajectory_label = torch.cat(
                    [trajectory_label[:, : config.trajectory_num_steps], labels.action],
                    dim=-1,
                )
            if trajectory_label is not None and labels.path is not None:
                # Path points follow the poses, zero-padded in the action columns.
                trajectory_label = torch.cat(
                    [
                        trajectory_label[:, : config.trajectory_num_steps],
                        F.pad(labels.path, (0, trajectory_label.shape[-1] - labels.path.shape[-1])),
                    ],
                    dim=1,
                )
            rows.append((self.planning_decoder, predictions.trajectory, trajectory_label))
        return rows

    @override
    def compute_loss(
        self,
        labels: TransfuserLabels,
        predictions: TransfuserPredictions,
    ) -> dict[str, torch.Tensor]:
        """Unweighted losses of every enabled head."""
        losses: dict[str, torch.Tensor] = {}
        for decoder, prediction, label in self._enabled_heads(labels, predictions):
            assert prediction is not None, f"{type(decoder).__name__} is enabled but its prediction is None"
            assert label is not None, f"{type(decoder).__name__} is enabled but its label is None"
            losses.update(decoder.compute_loss(prediction, label))
        return losses

    @override
    def compute_metrics(
        self,
        labels: TransfuserLabels,
        predictions: TransfuserPredictions,
    ) -> dict[str, torch.Tensor]:
        """Quality metrics of every enabled head."""
        metrics: dict[str, torch.Tensor] = {}
        for decoder, prediction, label in self._enabled_heads(labels, predictions):
            assert prediction is not None, f"{type(decoder).__name__} is enabled but its prediction is None"
            assert label is not None, f"{type(decoder).__name__} is enabled but its label is None"
            metrics.update(decoder.compute_metrics(prediction, label))
        return metrics

    @override
    def loss_weights(self, epoch: int) -> dict[str, float]:
        """Normalized loss weights; disabled heads get 0 so they leave the normalization sum."""
        del epoch
        config = self._config
        loss_config = config.loss_config
        weights = {
            "loss_semantic": loss_config.loss_weight_semantic if config.perspective_config.use_semantic else 0.0,
            "loss_depth": loss_config.loss_weight_depth if config.perspective_config.use_depth else 0.0,
            "loss_bev_semantic": loss_config.loss_weight_bev_semantic
            if config.bev_semantic_config.use_bev_semantic
            else 0.0,
            "loss_center_net_heatmap": loss_config.loss_weight_center_net_heatmap,
            "loss_center_net_wh": loss_config.loss_weight_center_net_wh,
            "loss_center_net_offset": loss_config.loss_weight_center_net_offset,
            "loss_center_net_yaw_class": loss_config.loss_weight_center_net_yaw_class,
            "loss_center_net_yaw_res": loss_config.loss_weight_center_net_yaw_res,
            "loss_center_net_velocity": loss_config.loss_weight_center_net_velocity,
            "loss_trajectory": loss_config.loss_weight_trajectory
            if config.planning_config.use_planning_decoder and config.planning_config.predict_waypoints
            else 0.0,
            "loss_path": loss_config.loss_weight_path
            if config.planning_config.use_planning_decoder and config.planning_config.path_config is not None
            else 0.0,
            "loss_action": loss_config.loss_weight_action
            if config.planning_config.use_planning_decoder and config.planning_config.unicycle_action_config is not None
            else 0.0,
        }
        if not config.box_detection_config.detect_boxes:
            for name in list(weights):
                if name.startswith("loss_center_net"):
                    weights[name] = 0.0
        if not config.box_detection_config.predict_box_velocity:
            weights["loss_center_net_velocity"] = 0.0
        return normalize_loss_weights(weights)

    @override
    def visualize_batch(
        self,
        features: TransfuserFeatures,
        labels: TransfuserLabels | None,
        navigation: NavigationConditioning,
        predictions: TransfuserPredictions,
        scene_api: SceneAPI,
    ) -> dict[str, torch.Tensor]:
        """Renders the prediction view, plus the label and BEV-map views when labels are given."""
        ego_state = scene_api.get_ego_state_se3_at_iteration(0)
        assert ego_state is not None, "Ego state should be available for visualization!"
        ego_box = ego_box_of(ego_state)
        route = (
            ego_frame_route_of(scene_api, self._config) if self._config.visualization_config.visualize_route else None
        )
        bev_semantic_classes = (
            bev_semantic_classes_of(predictions.bev_semantic, self._config)
            if predictions.bev_semantic is not None
            else None
        )
        images = {
            "predictions": render_prediction(
                features,
                navigation.target_points,
                predictions,
                self._config,
                bev_semantic_classes=bev_semantic_classes,
                route=route,
                ego_box=ego_box,
            ),
        }
        if labels is not None:
            images["labels"] = render_ground_truth(
                features,
                labels,
                navigation.target_points,
                self._config,
                route=route,
                boxes=ground_truth_boxes_of(scene_api, self._config),
                ego_box=ego_box,
            )
            if (
                predictions.boxes is not None
                or predictions.bev_semantic is not None
                or features.lidar_feature is not None
            ):
                images["bev_maps"] = render_bev_maps(
                    features,
                    labels,
                    predictions,
                    bev_semantic_classes=bev_semantic_classes,
                )
        return {
            name: torch.from_numpy(image)  # pyright: ignore[reportUnknownMemberType]
            for name, image in images.items()
        }
