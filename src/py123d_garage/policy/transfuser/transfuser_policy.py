# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

import torch
from py123d.api import SceneAPI
from typing_extensions import override

from py123d_garage.api.abstract_policy import (
    AbstractPolicy,
    CacheTensorSpec,
    normalize_loss_weights,
)
from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
from py123d_garage.cache.codec import (
    JpegCodec,
    PngCodec,
    ZlibCodec,
)
from py123d_garage.common.sensor.image_augmentation import augment_rgb_batch
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import (
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.features import (
    TransfuserFeatures,
    build_camera_feature,
    build_lidar_feature,
    build_velocity,
)
from py123d_garage.policy.transfuser.labels import (
    TransfuserLabels,
    build_bev_labels,
    build_perspective_labels,
    build_trajectory_label,
)
from py123d_garage.policy.transfuser.network.bev_semantic_decoder import (
    BEVSemanticDecoder,
    BevSemanticLabels,
)
from py123d_garage.policy.transfuser.network.center_net_decoder import (
    CenterNetBoundingBoxPrediction,
    CenterNetDecoder,
    CenterNetLabels,
)
from py123d_garage.policy.transfuser.network.perspective_decoder import (
    PerspectiveDecoder,
)
from py123d_garage.policy.transfuser.network.planning_decoder import (
    PlanningDecoder,
)
from py123d_garage.policy.transfuser.network.transfuser_backbone import (
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


@runtime_checkable
class _TaskHead(Protocol):
    def compute_loss(self, predictions: Any, label: Any, /) -> dict[str, torch.Tensor]: ...

    def compute_metrics(self, predictions: Any, label: Any, /) -> dict[str, torch.Tensor]: ...


class TransfuserPolicy(
    AbstractPolicy[TransfuserFeatures, TransfuserLabels, TransfuserPredictions],
):
    def __init__(self, config: PolicyConfig):
        """Initializes the TransFuser policy from its config block."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        assert config.transfuser_config is not None, (
            "the target names TransfuserPolicy but no transfuser_config block is declared"
        )
        self._config: TransfuserConfig = config.transfuser_config

    @property
    @override
    def policy_config(self) -> TransfuserConfig:
        return self._config

    @override
    def build_model(self, from_local_checkpoint: bool) -> None:
        config = self._config
        self.backbone = TransfuserBackbone(
            config,
            image_encoder_pretrained=config.backbone_config.image_encoder_pretrained and not from_local_checkpoint,
        )

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
            self.planning_decoder = PlanningDecoder(
                input_bev_channels=self.backbone.num_lidar_features,
                config=config,
            )

    @override
    def cache_signature(self, dataset: str) -> dict[str, dict[str, str]]:
        """Inherited; the camera entries narrow to the dataset's cameras."""
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
        config = self._config
        lidar_feature: torch.Tensor | None = None
        if not config.backbone_config.latent:
            lidar_feature = (
                cached["lidar_feature"] if "lidar_feature" in cached else build_lidar_feature(scene_api, config)
            )
        return TransfuserFeatures(
            camera_feature=cached["camera_feature"]
            if "camera_feature" in cached
            else build_camera_feature(scene_api, config),
            velocity=build_velocity(scene_api, config),
            lidar_feature=lidar_feature,
        )

    @override
    def build_labels(
        self,
        scene_api: SceneAPI,
        cached: dict[str, torch.Tensor],
    ) -> TransfuserLabels:
        config = self._config
        if cached:  # every label is cached
            center_net: CenterNetLabels | None = None
            if config.box_detection_config.detect_boxes:
                center_net = CenterNetLabels(
                    center_net_heatmap=cached["center_net_heatmap"],
                    center_net_wh=cached["center_net_wh"],
                    center_net_offset=cached["center_net_offset"],
                    center_net_yaw_class=cached["center_net_yaw_class"],
                    center_net_yaw_res=cached["center_net_yaw_res"],
                    center_net_pixel_weight=cached["center_net_pixel_weight"],
                    center_net_avg_factor=cached["center_net_avg_factor"],
                    center_net_velocity=cached["center_net_velocity"]
                    if config.box_detection_config.predict_box_velocity
                    else None,
                )
            return TransfuserLabels(
                trajectory=cached["trajectory"] if config.planning_config.use_planning_decoder else None,
                semantic=cached["semantic"] if config.perspective_config.use_semantic else None,
                depth=cached["depth"] if config.perspective_config.use_depth else None,
                bev_semantic=BevSemanticLabels(
                    bev_semantic=cached["bev_semantic"],
                    bev_semantic_has_map=cached["bev_semantic_has_map"],
                )
                if config.bev_semantic_config.use_bev_semantic
                else None,
                center_net=center_net,
            )
        trajectory: torch.Tensor | None = None
        if config.planning_config.use_planning_decoder:
            trajectory = build_trajectory_label(
                scene_api,
                config.trajectory_num_steps,
                config.trajectory_interval_us,
                config.planning_config.predict_yaw,
            )
        bev_semantic: BevSemanticLabels | None = None
        center_net = None
        if config.box_detection_config.detect_boxes or config.bev_semantic_config.use_bev_semantic:
            bev_semantic, center_net = build_bev_labels(scene_api, config)
        semantic: torch.Tensor | None = None
        depth: torch.Tensor | None = None
        if config.perspective_config.use_semantic or config.perspective_config.use_depth:
            semantic, depth = build_perspective_labels(scene_api, config)
        return TransfuserLabels(
            trajectory=trajectory,
            semantic=semantic,
            depth=depth,
            bev_semantic=bev_semantic,
            center_net=center_net,
        )

    @override
    def _cache_specs(self) -> dict[str, CacheTensorSpec]:
        """
        Returns for each tensor its codec and the config values its builder reads.

        Returns:
            spec by tensor name.
        """
        # NOTE@ln2697: only parameters are checked, not the builder implementation;
        # byte-changing builder edits need a manual force_cache_rebuild.
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
        }

        tensors: dict[str, CacheTensorSpec] = {
            # The stitched camera is a natural image and by far the largest
            # cached tensor; JPEG keeps the store to a fraction of a lossless one.
            "camera_feature": CacheTensorSpec(
                codec=JpegCodec(
                    quality=camera.cache_jpeg_quality,
                    quantization_scale=1,
                ),
                signature={
                    "input_cameras": input_cameras,
                    "image_width": str(camera.image_width),
                    "image_height": str(camera.image_height),
                    "cache_jpeg_quality": str(camera.cache_jpeg_quality),
                },
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
                },
            )
            tensors["bev_semantic_has_map"] = CacheTensorSpec(codec=ZlibCodec(), signature={})
        if config.planning_config.use_planning_decoder:
            tensors["trajectory"] = CacheTensorSpec(
                codec=ZlibCodec(),
                signature={
                    "trajectory_horizon_us": str(config.trajectory_horizon_us),
                    "trajectory_interval_us": str(config.trajectory_interval_us),
                    "predict_yaw": str(config.planning_config.predict_yaw),
                },
            )
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
                        # predict_box_velocity follows the accumulated sweeps.
                        "lidar_horizon_us": str(lidar.lidar_horizon_us),
                        "lidar_interval_us": str(lidar.lidar_interval_us),
                    },
                )
        if perspective.use_semantic:
            tensors["semantic"] = CacheTensorSpec(
                codec=PngCodec(quantization_scale=1),
                signature={**perspective_geometry},
            )
        if perspective.use_depth:
            tensors["depth"] = CacheTensorSpec(
                # 8-bit linear codes; build_perspective_labels asserts the dataset matches.
                codec=PngCodec(
                    quantization_scale=255.0 / perspective.depth_max_m,
                ),
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
    ) -> TransfuserFeatures:
        """
        Colour-augments the camera batch on device.

        Photometric only, so the LiDAR raster and metric inputs are left alone.
        """
        probability = self._config.camera_config.camera_augmentation_probability
        if probability <= 0.0:
            return features
        return replace(
            features,
            camera_feature=augment_rgb_batch(
                features.camera_feature,
                probability,
            ),
        )

    @override
    def forward(
        self,
        features: TransfuserFeatures,
        navigation: NavigationConditioning,
    ) -> TransfuserPredictions:
        """Runs the backbone and every config-enabled head."""
        config = self._config
        image = features.camera_feature
        lidar = None if config.backbone_config.latent else features.lidar_feature
        velocity = features.velocity if config.planning_conditioning_config.use_velocity else None

        bev_features, image_features = self.backbone(image, lidar)

        trajectory: torch.Tensor | None = None
        if config.planning_config.use_planning_decoder:
            trajectory = self.planning_decoder(
                bev_features,
                navigation.target_points,
                velocity,
            )
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
            labels: batched label bundle from build_labels.
            predictions: prediction bundle from forward.

        Returns:
            the enabled heads' decoders, each with its prediction and label.
        """
        config = self._config
        rows: list[tuple[_TaskHead, Any, Any]] = []
        if config.perspective_config.use_semantic:
            rows.append((self.semantic_decoder, predictions.semantic, labels.semantic))
        if config.perspective_config.use_depth:
            rows.append((self.depth_decoder, predictions.depth, labels.depth))
        if config.bev_semantic_config.use_bev_semantic:
            rows.append((self.bev_semantic_decoder, predictions.bev_semantic, labels.bev_semantic))
        if config.box_detection_config.detect_boxes:
            rows.append((self.center_net_decoder, predictions.boxes, labels.center_net))
        if config.planning_config.use_planning_decoder:
            rows.append((self.planning_decoder, predictions.trajectory, labels.trajectory))
        return rows

    @override
    def compute_loss(
        self,
        labels: TransfuserLabels,
        predictions: TransfuserPredictions,
    ) -> dict[str, torch.Tensor]:
        """Each decoder owns its unweighted loss."""
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
        """Each decoder owns its quality metrics, mirroring compute_loss."""
        metrics: dict[str, torch.Tensor] = {}
        for decoder, prediction, label in self._enabled_heads(labels, predictions):
            assert prediction is not None, f"{type(decoder).__name__} is enabled but its prediction is None"
            assert label is not None, f"{type(decoder).__name__} is enabled but its label is None"
            metrics.update(decoder.compute_metrics(prediction, label))
        return metrics

    @override
    def loss_weights(self, epoch: int) -> dict[str, float]:
        """
        Normalized loss weights for one epoch, with disabled heads zeroed out.

        A head that does not run produces no loss, so its weight must leave the
        normalization sum too.
        """
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
            if config.planning_config.use_planning_decoder
            else 0.0,
        }
        if not config.box_detection_config.detect_boxes:
            for name in list(weights):
                if name.startswith("loss_center_net"):
                    weights[name] = 0.0
        # The velocity head needs temporal LiDAR context; also unusable in latent mode.
        if not config.box_detection_config.predict_box_velocity or config.backbone_config.latent:
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
        """
        Renders the composed label/prediction views and the raw BEV-map grid.

        Without labels only the prediction view is rendered; the other two draw
        ground truth.
        """
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
