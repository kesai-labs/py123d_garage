from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from py123d.datatypes import (
    CameraID,
    DefaultBoxDetectionLabel,
    DefaultCameraSegmentationLabel,
    LidarID,
    MapLayer,
)
from typing_extensions import override

from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.config.schema.policy.visualization_config import VisualizationConfig
from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt
from py123d_garage.py123d_help.misc.constants import INPUT_LIDARS
from py123d_garage.py123d_help.scene_readers import past_offsets_us

# Total downsampling of the backbone's BEV branch; fixed by the architecture.
_BACKBONE_BEV_STRIDE = 32


@dataclass
class CameraConfig:
    """Camera selection, the stitched image geometry, and camera augmentation."""

    # Cameras the model ingests in left-to-right stitch order; for each dataset.
    input_cameras: dict[str, list[CameraID]] = field(default_factory=dict)
    # Final width of the stitched model input across the input cameras.
    image_width: int = 1024
    # Final height of the (stitched) model input image.
    image_height: int = 256
    # JPEG quality the cache store keeps the stitched image at;
    cache_jpeg_quality: int = 95
    # Per-sample, chance of each colour augmentation (noise, dropout) on the camera batch
    # during training; 0.0 = no augmentation.
    camera_augmentation_probability: float = 0.0

    @property
    def img_vert_anchors(self) -> int:
        """Number of vertical anchors for image feature maps."""
        return self.image_height // _BACKBONE_BEV_STRIDE

    @property
    def img_horz_anchors(self) -> int:
        """Number of horizontal anchors for image feature maps."""
        return self.image_width // _BACKBONE_BEV_STRIDE


@dataclass
class LidarConfig:
    """Point filtering, the sweep window, and the BEV raster and feature-map geometry."""

    # --- BEV raster the network sees ---
    # Pixels per meter of the BEV raster; with the extents below, sets the token count.
    bev_pixels_per_meter: float = 4.0
    # Back boundary of the BEV crop in meters (ego frame, x points forward).
    bev_min_x_m: float = -32.0
    # Front boundary of the BEV crop in meters.
    bev_max_x_m: float = 64.0
    # Right boundary of the BEV crop in meters (y points left).
    bev_min_y_m: float = -40.0
    # Left boundary of the BEV crop in meters.
    bev_max_y_m: float = 40.0

    # --- Point filtering ---
    # Number of channels of the rasterized LiDAR input (e.g. 2 with a ground-plane channel).
    lidar_in_channels: int = 1
    # Max number of LiDAR points per pixel in the rasterized LiDAR splat.
    max_lidar_points_per_bev_pixel: int = 5
    # Maximum height threshold for LiDAR points (meters, points above are discarded).
    lidar_max_height_m: float = 10.0
    # Minimum height threshold for LiDAR points (meters, points below are discarded).
    lidar_min_height_m: float = -4.0
    # If true remove ground points from the LiDAR input, by the method below.
    remove_lidar_ground_points: bool = True
    # "fixed_z": height cut at lidar_ground_z_m.
    # "fitted_plane": per-segment fitted ground planes that follow the road (~10 ms/sweep).
    ground_removal: str = "fitted_plane"
    # Height cut used by ground_removal="fixed_z" (meters above the rear axle).
    lidar_ground_z_m: float = 0.2

    # --- Sweep window ---
    # Target spacing of the stacked past sweeps, and half of it the matching
    # tolerance around each target timestamp; required with a horizon.
    lidar_interval_us: PositiveInt | None = None
    # How far back the stacked past sweeps reach; 0 stacks the anchor sweep only.
    lidar_horizon_us: NonNegativeInt = 0

    @property
    def bev_width_pixel(self) -> int:
        """Width of the BEV raster in pixels (x axis)."""
        return int(
            (self.bev_max_x_m - self.bev_min_x_m) * self.bev_pixels_per_meter,
        )

    @property
    def bev_height_pixel(self) -> int:
        """Height of the BEV raster in pixels (y axis)."""
        return int(
            (self.bev_max_y_m - self.bev_min_y_m) * self.bev_pixels_per_meter,
        )

    @property
    def lidar_bev_grid_rows(self) -> int:
        """Number of vertical anchors (rows) of the BEV token grid."""
        return self.bev_height_pixel // _BACKBONE_BEV_STRIDE

    @property
    def lidar_bev_grid_cols(self) -> int:
        """Number of horizontal anchors (columns) of the BEV token grid."""
        return self.bev_width_pixel // _BACKBONE_BEV_STRIDE


@dataclass
class BackboneConfig:
    """Image/LiDAR encoders and the GPT fusion layers."""

    # Train with the image and LiDAR encoders frozen.
    freeze_backbone: bool = False
    # "bilinear" or "nearest" for the fusion and perspective up-samples; the top-down and BEV
    # heads stay bilinear either way.
    upsample_mode: str = "bilinear"
    # Architecture name for image encoder backbone (timm model name).
    image_architecture: str = "resnet34"
    # Architecture name for LiDAR encoder backbone (timm model name).
    lidar_architecture: str = "resnet34"
    # Start from timm's pretrained image-encoder weights.
    image_encoder_pretrained: bool = True
    # Latent TransFuser: replace the LiDAR raster with a 2-channel positional-encoding
    # grid, turning the model into a camera-only architecture with the same BEV heads.
    latent: bool = False

    # --- GPT encoder (cross-modal fusion transformers) ---
    # MLP expansion factor inside each transformer block.
    block_exp: int = 4
    # Transformer layers per fusion stage.
    n_layer: int = 2
    # Attention heads per transformer layer.
    n_head: int = 4
    # Dropout on the token embeddings.
    embd_pdrop: float = 0.1
    # Dropout on the residual branches.
    resid_pdrop: float = 0.1
    # Dropout on the attention weights.
    attn_pdrop: float = 0.1
    # Normal-init mean of the linear layers.
    gpt_linear_layer_init_mean: float = 0.0
    # Normal-init std of the linear layers.
    gpt_linear_layer_init_std: float = 0.02
    # Initial weight of the LayerNorms.
    gpt_layer_norm_init_weight: float = 1.0


@dataclass
class PlanningConditioningConfig:
    """Ego-state and target-point network inputs; the distances live on the policy contract."""

    # Feed the ego velocity to the planning decoder.
    use_velocity: bool = True
    # Maximum speed limit for the vehicle in m/s (velocity normalization constant).
    max_speed_mps: float = 25.0
    # Normalization constants for target points [x_norm, y_norm] in meters.
    target_points_normalization_constants: list[float] = field(
        default_factory=lambda: [50.0, 50.0],
    )


@dataclass
class PlanningConfig:
    """Planning decoder and waypoint prediction."""

    # Predict waypoints with the BEV cross-attention planning decoder.
    use_planning_decoder: bool = True
    # Predict a yaw per waypoint instead of deriving it from the path.
    predict_yaw: bool = True
    # Cross-attention layers of the planning decoder.
    transfuser_num_bev_cross_attention_layers: int = 6
    # Attention heads per cross-attention layer.
    transfuser_num_bev_cross_attention_heads: int = 8
    # Token width of the planning decoder.
    transfuser_token_dim: int = 256


@dataclass
class BevSemanticConfig:
    """BEV semantic-segmentation auxiliary task."""

    # Predict BEV semantic segmentation.
    use_bev_semantic: bool = True
    # Semantic index -> (geometry kind, py123d entities); index 0 is the implicit background.
    # "polygon"/"linestring" rasterize map layers, "box" the listed detection labels.
    bev_semantic_classes: ClassVar[dict[int, tuple[str, list[MapLayer | DefaultBoxDetectionLabel]]]] = {
        1: (
            "polygon",
            [MapLayer.LANE, MapLayer.INTERSECTION, MapLayer.GENERIC_DRIVABLE],
        ),  # road
        2: ("polygon", [MapLayer.WALKWAY]),  # walkways
        3: ("linestring", [MapLayer.LANE]),  # centerline
        4: (
            "box",
            [
                DefaultBoxDetectionLabel.TRAFFIC_SIGN,
                DefaultBoxDetectionLabel.BARRIER,
                DefaultBoxDetectionLabel.TRAFFIC_CONE,
                DefaultBoxDetectionLabel.GENERIC_OBJECT,
            ],
        ),  # static objects
        5: ("box", [DefaultBoxDetectionLabel.VEHICLE]),  # vehicles
        6: ("box", [DefaultBoxDetectionLabel.PERSON]),  # pedestrians
    }
    # Keys of bev_semantic_classes to predict; the label uses 1, 2, ... in this order.
    bev_semantic_class_ids: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6],
    )
    # Inflate pedestrian boxes in the BEV labels by this factor.
    pedestrian_bev_extent_scale: float = 1.0
    # Lower bound on a rasterized pedestrian's full extent (length and width each).
    pedestrian_bev_min_extent_m: float = 0.4
    # Channels of the BEV feature map feeding the head.
    bev_feature_channels: int = 64
    # Label raster down-sampling relative to the BEV raster.
    bev_downsample_factor: int = 4
    # Up-sampling factor inside the BEV head.
    bev_upsample_factor: int = 2

    @property
    def selected_bev_semantic_classes(
        self,
    ) -> dict[int, tuple[str, list[MapLayer | DefaultBoxDetectionLabel]]]:
        """The selected bev_semantic_classes entries, keyed 1, 2, ... in selection order."""
        unknown = set(self.bev_semantic_class_ids) - set(self.bev_semantic_classes)
        if unknown:
            raise ValueError(f"bev_semantic_class_ids {sorted(unknown)} are no bev_semantic_classes keys.")
        return {
            index + 1: self.bev_semantic_classes[class_id] for index, class_id in enumerate(self.bev_semantic_class_ids)
        }

    @property
    def num_bev_semantic_classes(self) -> int:
        """Total number of BEV semantic segmentation classes, incl. background."""
        return len(self.bev_semantic_class_ids) + 1


@dataclass
class BoxDetectionConfig:
    """CenterNet bounding-box auxiliary task."""

    # Predict bounding boxes.
    detect_boxes: bool = True
    # Detection class definition: class index -> py123d box labels grouped into that class.
    detection_classes: dict[int, list[DefaultBoxDetectionLabel]] = field(
        default_factory=lambda: {
            0: [DefaultBoxDetectionLabel.VEHICLE],  # vehicles
            1: [DefaultBoxDetectionLabel.PERSON],  # pedestrians
            2: [DefaultBoxDetectionLabel.TWO_WHEELER],  # bikers
            3: [
                DefaultBoxDetectionLabel.TRAFFIC_SIGN,
                DefaultBoxDetectionLabel.BARRIER,
                DefaultBoxDetectionLabel.TRAFFIC_CONE,
                DefaultBoxDetectionLabel.GENERIC_OBJECT,
            ],  # static obstacles
        },
    )
    # Keep predicted boxes above this confidence.
    box_confidence_threshold: float = 0.3
    # Max boxes in the training labels.
    max_num_boxes: int = 90
    # Discretization bins for the yaw prediction.
    num_yaw_bins: int = 12
    # Max peaks kept from the CenterNet heatmap.
    max_center_net_detections: int = 100
    # Kernel of the heatmap max-pooling NMS.
    center_net_max_pooling_kernel_size: int = 3
    # Channels of the BEV feature map feeding the box head.
    box_head_input_channels: int = 64
    # Velocity head of the box detection.
    predict_box_velocity: bool = False

    @property
    def num_box_classes(self) -> int:
        """Total number of bounding box classes to detect."""
        return len(self.detection_classes)


@dataclass
class PerspectiveConfig:
    """Semantic-segmentation and depth auxiliary tasks (default off: they need camera semantic/depth streams)."""

    # Predict perspective semantic segmentation.
    use_semantic: bool = False
    # The labels are the dataset's classes converted to py123d's default taxonomy.
    num_semantic_classes: int = len(DefaultCameraSegmentationLabel)
    # Resolution at which the perspective auxiliary tasks are predicted.
    perspective_downsample_factor: int = 1
    # Channels after the first up-sample.
    deconv_channel_num_0: int = 128
    # Channels after the second up-sample.
    deconv_channel_num_1: int = 64
    # Channels after the third up-sample.
    deconv_channel_num_2: int = 32
    # Fraction of the down-sampling factor that will be up-sampled in the first Up-sample.
    deconv_scale_factor_0: int = 4
    # Fraction of the down-sampling factor that will be up-sampled in the second Up-sample.
    deconv_scale_factor_1: int = 8

    # --- Depth ---
    # Predict perspective depth.
    use_depth: bool = False
    # Far plane of the depth labels and visualization colormap; must equal the
    # dataset's depth-quantization far plane.
    depth_max_m: float = 50.0


@dataclass
class TransfuserVisualizationConfig(VisualizationConfig):
    """The shared overlay styles plus the map route TransFuser draws."""

    # Draws the map route the target points are sampled from into every rendered view.
    visualize_route: bool = True
    # False draws predicted map classes as background; for models trained without maps.
    visualize_bev_map_classes: bool = True
    # The route polyline, tab10 orange.
    route_color_rgb: list[int] = field(default_factory=lambda: [255, 127, 14])
    # Line width of the drawn route.
    route_thickness_pixel: int = 1


@dataclass
class LossConfig:
    """Loss weights, unnormalized; the trainer normalizes across enabled heads."""

    loss_weight_semantic: float = 1.0
    loss_weight_depth: float = 0.00001
    loss_weight_bev_semantic: float = 1.0
    loss_weight_center_net_heatmap: float = 1.0
    loss_weight_center_net_wh: float = 1.0
    loss_weight_center_net_offset: float = 1.0
    loss_weight_center_net_yaw_class: float = 1.0
    loss_weight_center_net_yaw_res: float = 1.0
    loss_weight_center_net_velocity: float = 1.0
    loss_weight_trajectory: float = 1.0


@dataclass
class TransfuserConfig(AbstractPolicyConfig):
    """
    TransFuser architecture, auxiliary heads, and network-input knobs.

    Values that cross sections (the signature, cross-section derivations) live
    here; everything else sits in its section's config object.
    """

    # -- Config objects --

    # Stitched image geometry and camera augmentation.
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    # Point filtering, sweep window, and BEV raster geometry.
    lidar_config: LidarConfig = field(default_factory=LidarConfig)
    # Image/LiDAR encoders and the GPT fusion layers.
    backbone_config: BackboneConfig = field(default_factory=BackboneConfig)
    # Target points and ego-state network inputs.
    planning_conditioning_config: PlanningConditioningConfig = field(
        default_factory=PlanningConditioningConfig,
    )
    # Planning decoder and waypoint prediction.
    planning_config: PlanningConfig = field(default_factory=PlanningConfig)
    # BEV semantic-segmentation auxiliary task.
    bev_semantic_config: BevSemanticConfig = field(default_factory=BevSemanticConfig)
    # CenterNet bounding-box auxiliary task.
    box_detection_config: BoxDetectionConfig = field(
        default_factory=BoxDetectionConfig,
    )
    # Perspective semantic-segmentation and depth auxiliary tasks.
    perspective_config: PerspectiveConfig = field(
        default_factory=PerspectiveConfig,
    )
    # Loss weights of the heads.
    loss_config: LossConfig = field(default_factory=LossConfig)
    # Options of the rendered training views.
    visualization_config: TransfuserVisualizationConfig = field(
        default_factory=TransfuserVisualizationConfig,
    )

    def __post_init__(self) -> None:
        sweep_offsets_us = past_offsets_us(
            self.lidar_config.lidar_horizon_us,
            self.lidar_config.lidar_interval_us,
        )
        if self.box_detection_config.predict_box_velocity and not sweep_offsets_us:
            raise ValueError("predict_box_velocity needs motion cues: set lidar_horizon_us.")

    @property
    @override
    def required_cameras(self) -> dict[str, list[CameraID]]:
        return self.camera_config.input_cameras

    @property
    @override
    def required_lidars(self) -> dict[str, list[LidarID]]:
        if self.backbone_config.latent:
            return {}
        return INPUT_LIDARS

    @property
    @override
    def required_history_duration_us(self) -> NonNegativeInt:
        return self.lidar_config.lidar_horizon_us

    @property
    @override
    def required_past_lidar_interval_us(self) -> PositiveInt | None:
        if not self.lidar_config.lidar_horizon_us:
            return None
        return self.lidar_config.lidar_interval_us
