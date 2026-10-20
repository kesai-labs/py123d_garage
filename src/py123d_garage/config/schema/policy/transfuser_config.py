# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

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
from py123d_garage.config.schema.policy.visualization_config import TrajectoryStyleConfig, VisualizationConfig
from py123d_garage.datatypes.numerics import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from py123d_garage.py123d_help.misc.constants import INPUT_LIDARS
from py123d_garage.py123d_help.scene_readers import past_offsets_us

# Total down-sampling of the backbone feature maps; fixed by the architecture.
_BACKBONE_BEV_STRIDE = 32


@dataclass
class CameraConfig:
    """Camera selection and the stitched image geometry."""

    # Cameras per dataset, in left-to-right stitch order.
    input_cameras: dict[str, list[CameraID]] = field(default_factory=dict)
    # Width of the stitched input image, in pixels.
    image_width: int = 1024
    # Height of the stitched input image, in pixels.
    image_height: int = 256
    # JPEG quality of the stitched image in the cache store.
    cache_jpeg_quality: int = 95
    # Time between the past camera rows stacked above the current one; 0 uses the current frame only.
    past_camera_offset_us: NonNegativeInt = 0
    # Number of past camera rows; every row gets an equal share of image_height.
    num_past_camera_rows: PositiveInt = 1

    @property
    def front_camera_index(self) -> int:
        """
        Stitch position of the _F0 front camera, which must be the same for every dataset.

        Raises:
            ValueError: when a dataset has no front camera or the datasets place it differently.
        """
        positions = {
            dataset: [index for index, camera_id in enumerate(camera_ids) if camera_id.name.endswith("_F0")]
            for dataset, camera_ids in self.input_cameras.items()
        }
        distinct = {tuple(found) for found in positions.values()}
        if len(distinct) > 1 or any(len(found) != 1 for found in positions.values()):
            raise ValueError(f"every dataset needs one front (_F0) camera at the same position, got {positions}")
        return next(iter(distinct))[0] if distinct else 0

    @property
    def img_vert_anchors(self) -> int:
        """Rows of the image feature map."""
        return self.image_height // _BACKBONE_BEV_STRIDE

    @property
    def img_horz_anchors(self) -> int:
        """Columns of the image feature map."""
        return self.image_width // _BACKBONE_BEV_STRIDE


@dataclass
class LidarConfig:
    """Point filtering, the sweep window and the BEV raster geometry."""

    # --- BEV raster ---
    # Resolution of the BEV raster, in pixels per meter.
    bev_pixels_per_meter: float = 4.0
    # Back edge of the BEV crop in the ego frame, in meters; x points forward.
    bev_min_x_m: float = -32.0
    # Front edge of the BEV crop, in meters.
    bev_max_x_m: float = 64.0
    # Right edge of the BEV crop, in meters; y points left.
    bev_min_y_m: float = -40.0
    # Left edge of the BEV crop, in meters.
    bev_max_y_m: float = 40.0

    # --- Point filtering ---
    # Channels of the LiDAR raster; only 1 is supported.
    lidar_in_channels: int = 1
    # Point count per BEV pixel that maps to the full raster value; higher counts are clipped.
    max_lidar_points_per_bev_pixel: int = 5
    # Drop points above this height, in meters.
    lidar_max_height_m: float = 10.0
    # Drop points below this height, in meters.
    lidar_min_height_m: float = -4.0
    # Remove ground points with the method in ground_removal.
    remove_lidar_ground_points: bool = True
    # "fixed_z" cuts at lidar_ground_z_m; "fitted_plane" fits a ground plane per segment (~10 ms per sweep).
    ground_removal: str = "fitted_plane"
    # Ground cut height above the rear axle for ground_removal="fixed_z", in meters.
    lidar_ground_z_m: float = 0.2

    # --- Sweep window ---
    # Spacing of the past sweeps, matched within half of it; required when lidar_horizon_us > 0.
    lidar_interval_us: PositiveInt | None = None
    # How far back the stacked past sweeps reach; 0 uses the current sweep only.
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
        """Rows of the BEV token grid."""
        return self.bev_height_pixel // _BACKBONE_BEV_STRIDE

    @property
    def lidar_bev_grid_cols(self) -> int:
        """Columns of the BEV token grid."""
        return self.bev_width_pixel // _BACKBONE_BEV_STRIDE


@dataclass
class BackboneConfig:
    """Image and LiDAR encoders and the GPT fusion layers."""

    # Freeze the backbone in eval mode; the heads and the planning decoder still train.
    freeze_backbone: bool = False
    # Interpolation mode of the feature map up-sampling; "bilinear_matmul" equals "bilinear" with a faster backward.
    upsample_mode: str = "bilinear_matmul"
    # timm model name of the image encoder.
    image_architecture: str = "resnet34"
    # timm model name of the LiDAR encoder.
    lidar_architecture: str = "resnet34"
    # Start the image encoder from timm's pretrained weights.
    image_encoder_pretrained: bool = True
    # Replace the LiDAR raster with a positional-encoding grid, which makes the model camera-only.
    latent: bool = False
    # "module:Class" path of the backbone; the class takes (config, image_encoder_pretrained).
    backbone_target: str = "py123d_garage.policy.transfuser.network.encoder.transfuser:TransfuserBackbone"
    # "module:function" patches that swap custom kernels into the backbone in training, keeping the state_dict keys.
    kernel_patches: list[str] = field(default_factory=list)

    # --- GPT fusion transformers ---
    # MLP expansion factor of each transformer block.
    block_exp: int = 4
    # Transformer layers per fusion stage.
    n_layer: int = 2
    # One value for each of the four fusion stages; None uses n_layer for all.
    n_layer_per_stage: list[int] | None = None
    # Attention heads per transformer layer.
    n_head: int = 4
    # One value for each of the four fusion stages; None uses n_head for all.
    n_head_per_stage: list[int] | None = None
    # Dropout on the token embeddings.
    embd_pdrop: float = 0.1
    # Dropout on the residual branches.
    resid_pdrop: float = 0.1
    # Dropout on the attention weights.
    attn_pdrop: float = 0.1
    # Register tokens in the last fusion stage; only the planning decoder reads them.
    num_fusion_registers: int = 8
    # Mean of the normal init of the linear layers.
    gpt_linear_layer_init_mean: float = 0.0
    # Std of the normal init of the linear layers.
    gpt_linear_layer_init_std: float = 0.02
    # Initial weight of the LayerNorms.
    gpt_layer_norm_init_weight: float = 1.0


@dataclass
class PlanningConditioningConfig:
    """Ego-state and target-point inputs of the network."""

    # Feed the ego speed as a scalar.
    use_velocity: bool = True
    # Feed the ego velocity vector, in m/s.
    use_velocity_vector: bool = False
    # Feed the ego acceleration vector, in m/s^2.
    use_acceleration_vector: bool = False
    # How far back the past ego poses reach; 0 turns them off.
    ego_history_duration_us: NonNegativeInt = 0
    # Spacing of the past ego poses; the sources store ego states every 100 ms.
    ego_history_interval_us: PositiveInt = 100_000
    # Width of the sine embedding of each past pose value.
    ego_history_embedding_dim: PositiveInt = 32
    # Put all past poses in one token instead of one token per step.
    ego_history_single_token: bool = False
    # Speed in m/s that normalizes the speed input.
    max_speed_mps: float = 45.0
    # Normalizes v_x and v_y, in m/s.
    velocity_normalization_constants: list[float] = field(
        default_factory=lambda: [45.0, 2.0],
    )
    # Normalizes a_x and a_y, in m/s^2.
    acceleration_normalization_constants: list[float] = field(
        default_factory=lambda: [4.0, 4.0],
    )
    # Normalizes the target point x and y, in meters.
    target_points_normalization_constants: list[float] = field(
        default_factory=lambda: [50.0, 50.0],
    )
    # Chance to replace each target point with NaN, which means no route intent; 1.0 drops all of them.
    target_point_dropout_probability: float = 0.0
    # Training-only dropout chance per target point; when set, target_point_dropout_probability must stay 0.0.
    target_point_dropout_probabilities: list[float] | None = None
    # Target points, by distance, fed at inference, the rest read as dropped; None feeds all.
    test_time_target_point_distances_m: list[float] | None = None
    # First training epochs that drop every target point.
    unconditional_epochs: NonNegativeInt = 0
    # Training-only Gaussian shift of each target point along the route's left normal.
    target_point_lateral_noise_std_m: float = 0.0
    # One lateral shift per sample for every target point, all to the same side, instead of one per point.
    target_point_lateral_noise_correlated: bool = False
    # Training-only relative Gaussian scaling of the speed input.
    speed_relative_noise_std: float = 0.0
    # Chance per sample of a learned null token in place of the speed input; 1.0 also drops it at inference.
    speed_dropout_probability: float = 0.0


@dataclass
class FlowMatchingConfig:
    """Flow-matching sampler and training noise of the planning decoder."""

    # Euler steps from noise to the trajectory.
    num_inference_steps: PositiveInt = 8
    # Noise draws per backbone pass in training.
    num_noise_draws: PositiveInt = 4
    # Project a sample's context tokens once for all its noise draws instead of once per draw; faster, same math.
    shared_draw_context: bool = True
    # Seed of the inference noise, which is reused at every replan.
    noise_seed: int = 0
    # Scale of the initial inference noise; 0 starts from the zero state, so the plan does not depend on the seed.
    noise_temperature: NonNegativeFloat = 0.0
    # Draw new noise at every sample instead of reusing noise_seed's.
    resample_noise: bool = False
    # Trajectories averaged per plan, each from its own noise.
    num_averaged_samples: PositiveInt = 1
    # Training noise levels follow Beta(alpha, 1); 1 is uniform.
    noise_level_beta_alpha: PositiveFloat = 1.0
    # Output that drives the plan at inference: "waypoints", "actions", "waypoints_on_path" or "actions_on_path".
    driven_output: str = "waypoints"
    # False stops the waypoints from attending to the actions.
    waypoints_attend_to_actions: bool = True
    # False stops the actions from attending to the waypoints.
    actions_attend_to_waypoints: bool = True
    # False hides the target points from the waypoints and actions; only the path reads them.
    waypoints_and_actions_attend_to_target_points: bool = True


@dataclass
class ActionConfig:
    """Unicycle actions of the planning decoder: normalization, bounds and loss terms."""

    # Mean used to normalize the acceleration action, in m/s^2.
    accel_mean_mps2: float = 0.0
    # Std used to normalize the acceleration action, in m/s^2.
    accel_std_mps2: PositiveFloat = 0.68
    # Mean used to normalize the curvature action, in 1/m.
    curvature_mean_per_m: float = 0.0
    # Std used to normalize the curvature action, in 1/m.
    curvature_std_per_m: PositiveFloat = 0.026
    # L1 between the rollout of the predicted actions and the logged poses.
    rollout_pose_loss: bool = True
    # L1 on the normalized actions.
    action_loss: bool = False
    # L1 between the speeds reached by the predicted and by the label accelerations.
    rollout_speed_loss: bool = True
    # Predict lateral acceleration (curvature times speed squared, speed at least 5 m/s) instead of curvature.
    lateral_accel_actions: bool = True
    # Std used to normalize the lateral acceleration action, in m/s^2.
    lateral_accel_std_mps2: PositiveFloat = 0.675
    # Squash the actions into the AlpaSim MPC limits below before the rollout.
    bounded: bool = True
    # Largest braking of the bounded actions, in m/s^2.
    max_brake_mps2: PositiveFloat = 9.0
    # Largest forward acceleration of the bounded actions, in m/s^2.
    max_accel_mps2: PositiveFloat = 6.0
    # Largest lateral acceleration of the bounded actions, in m/s^2.
    max_lateral_accel_mps2: PositiveFloat = 7.0


@dataclass
class PathConfig:
    """Route points evenly spaced ahead, predicted next to the waypoints."""

    # Number of path points.
    num_points: PositiveInt = 64
    # Distance between neighbouring path points along the route, in meters.
    spacing_m: PositiveFloat = 1.0
    # False stops the path from attending to the waypoints; the waypoints still attend to the path.
    path_attends_to_waypoints: bool = True
    # False stops the path from attending to the actions; the actions still attend to the path.
    path_attends_to_actions: bool = True
    # Target points, by distance, the path can see; None means all.
    visible_target_point_distances_m: list[float] | None = None


@dataclass
class PlanningConfig:
    """Planning decoder and waypoint prediction."""

    # Build the planning decoder, which predicts the waypoints.
    use_planning_decoder: bool = True
    # Layers of the planning decoder.
    transfuser_num_bev_cross_attention_layers: int = 6
    # Attention heads per planning decoder layer.
    transfuser_num_bev_cross_attention_heads: int = 8
    # Token width of the planning decoder.
    transfuser_token_dim: int = 256
    # Gate each planning decoder attention output by a sigmoid of its queries.
    gated_attention: bool = False
    # "module:Class" path of the planning decoder; the regression decoder needs no actions and no path config.
    planning_decoder_target: str = (
        "py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder:FlowPlanningDecoder"
    )
    # Flow-matching sampler of the planning decoder.
    flow_config: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    # Predict the waypoints; False (flow decoder only) leaves the action rollout to drive.
    predict_waypoints: bool = True
    # Unicycle actions, with labels fitted to the trajectory; None skips them.
    unicycle_action_config: ActionConfig | None = field(default_factory=ActionConfig)
    # Path points, with labels taken from the log's route; None skips them.
    path_config: PathConfig | None = field(default_factory=PathConfig)
    # "module:function" patches that swap fused kernels into the planning decoder in training, not in evaluation.
    kernel_patches: list[str] = field(default_factory=list)


@dataclass
class BevSemanticConfig:
    """BEV semantic-segmentation auxiliary task."""

    # Predict BEV semantic segmentation.
    use_bev_semantic: bool = True
    # Class index -> (geometry kind, py123d entities); index 0 is the background.
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
    # Keys of bev_semantic_classes to predict; the labels use 1, 2, ... in this order.
    bev_semantic_class_ids: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6],
    )
    # Scale factor on the size of pedestrian boxes in the BEV labels.
    pedestrian_bev_extent_scale: float = 1.0
    # Minimum full length and width of a drawn pedestrian, in meters.
    pedestrian_bev_min_extent_m: float = 0.4
    # Skip map objects whose nearest vertex is further than this from the ego height; None draws all.
    map_z_gate_m: float | None = None
    # Channels of the BEV feature map that feeds the head.
    bev_feature_channels: int = 64
    # Down-sampling of the label raster relative to the BEV raster.
    bev_downsample_factor: int = 4
    # Up-sampling factor of the first up-sample of the BEV feature map.
    bev_upsample_factor: int = 2
    # "module:function" patches that swap custom kernels into the BEV semantic decoder in training, not in evaluation.
    kernel_patches: list[str] = field(default_factory=list)
    # Compute the loss as logsumexp over all classes minus logsumexp over the allowed ones, without the log-softmax
    # tensor: the same loss in fewer passes over the logits.
    two_logsumexp_loss: bool = True

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
        """Number of BEV semantic classes, including the background."""
        return len(self.bev_semantic_class_ids) + 1


@dataclass
class BoxDetectionConfig:
    """CenterNet bounding-box auxiliary task."""

    # Predict bounding boxes.
    detect_boxes: bool = True
    # Class index -> py123d box labels merged into that class.
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
    # Number of bins of the yaw classification.
    num_yaw_bins: int = 12
    # Max peaks kept from the CenterNet heatmap.
    max_center_net_detections: int = 100
    # Kernel size of the max-pooling that suppresses non-peak heatmap cells.
    center_net_max_pooling_kernel_size: int = 3
    # Channels of the BEV feature map that feeds the box head.
    box_head_input_channels: int = 64
    # Predict box velocities; needs motion cues in the input, see has_motion_cues.
    predict_box_velocity: bool = False
    # "module:function" patches that swap custom kernels into the CenterNet decoder in training, not in evaluation.
    kernel_patches: list[str] = field(default_factory=list)

    @property
    def num_box_classes(self) -> int:
        """Number of box classes."""
        return len(self.detection_classes)


@dataclass
class PerspectiveConfig:
    """Perspective semantic-segmentation and depth auxiliary tasks; they need camera semantic and depth streams."""

    # Predict perspective semantic segmentation.
    use_semantic: bool = False
    # Dataset class ids to keep, in order; None maps to py123d's default taxonomy.
    semantic_class_ids: list[int] | None = None
    # Number of semantic classes; must match semantic_class_ids, or the default taxonomy, minus the merged classes.
    num_semantic_classes: int = len(DefaultCameraSegmentationLabel)
    # Class id -> id it merges into, applied at read time so the cache keeps every class.
    semantic_class_merges: dict[int, int] = field(default_factory=dict)
    # Class index left out of the semantic loss and metrics.
    semantic_ignore_index: int | None = None
    # Down-sampling of the perspective predictions and labels relative to the input image.
    perspective_downsample_factor: int = 1
    # Channels after the first up-sample.
    deconv_channel_num_0: int = 128
    # Channels after the second up-sample.
    deconv_channel_num_1: int = 64
    # Channels after the third up-sample.
    deconv_channel_num_2: int = 32
    # Divisor of the total up-sampling that gives the first up-sample's scale.
    deconv_scale_factor_0: int = 4
    # Divisor of the total up-sampling that gives the second up-sample's scale.
    deconv_scale_factor_1: int = 8

    # --- Depth ---
    # Predict perspective depth.
    use_depth: bool = False
    # Far plane of the depth labels, in meters; must equal the dataset's depth-quantization far plane.
    depth_max_m: float = 50.0


@dataclass
class TransfuserVisualizationConfig(VisualizationConfig):
    """The shared overlay styles plus the map route."""

    # Draw the map route the target points come from into every rendered view.
    visualize_route: bool = True
    # Draw the predicted map classes; false draws them as background, for models trained without maps.
    visualize_bev_map_classes: bool = True
    # Color of the drawn route, as RGB; red like the route in the AlpaSim view.
    route_color_rgb: list[int] = field(default_factory=lambda: [214, 39, 40])
    # Line width of the drawn route, in pixels.
    route_thickness_pixel: int = 3
    # Style of the predicted path, drawn as a line under the trajectory dots.
    prediction_path_style: TrajectoryStyleConfig = field(
        default_factory=lambda: TrajectoryStyleConfig(
            color_rgb=[44, 160, 44],
            line_thickness_pixel=3,
            pose_radius_pixel=0,
        ),
    )
    # Style of the label path points.
    ground_truth_path_style: TrajectoryStyleConfig = field(
        default_factory=lambda: TrajectoryStyleConfig(
            color_rgb=[152, 223, 138],
            line_thickness_pixel=2,
            pose_radius_pixel=0,
        ),
    )


@dataclass
class LossConfig:
    """Loss weights; the trainer normalizes them over the enabled heads."""

    # Weight of the perspective semantic segmentation loss.
    loss_weight_semantic: float = 1.0
    # Weight of the perspective depth loss.
    loss_weight_depth: float = 0.00001
    # Weight of the BEV semantic segmentation loss.
    loss_weight_bev_semantic: float = 1.0
    # Weight of the box center heatmap loss.
    loss_weight_center_net_heatmap: float = 1.0
    # Weight of the box width and length loss.
    loss_weight_center_net_wh: float = 1.0
    # Weight of the box sub-pixel center offset loss.
    loss_weight_center_net_offset: float = 1.0
    # Weight of the box yaw bin loss.
    loss_weight_center_net_yaw_class: float = 1.0
    # Weight of the box yaw residual loss inside the bin.
    loss_weight_center_net_yaw_res: float = 1.0
    # Weight of the box velocity loss.
    loss_weight_center_net_velocity: float = 1.0
    # Weight of the waypoint loss of the planning decoder.
    loss_weight_trajectory: float = 1.0
    # Weight of the path point loss of the planning decoder.
    loss_weight_path: float = 1.0
    # Weight of the unicycle action loss of the planning decoder.
    loss_weight_action: float = 1.0


@dataclass
class TransfuserConfig(AbstractPolicyConfig):
    """TransFuser architecture, auxiliary heads and network inputs."""

    # Camera selection and the stitched image geometry.
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    # Point filtering, the sweep window and the BEV raster geometry.
    lidar_config: LidarConfig = field(default_factory=LidarConfig)
    # Image and LiDAR encoders and the GPT fusion layers.
    backbone_config: BackboneConfig = field(default_factory=BackboneConfig)
    # Ego-state and target-point inputs of the network.
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
    # Options of the rendered views.
    visualization_config: TransfuserVisualizationConfig = field(
        default_factory=TransfuserVisualizationConfig,
    )

    def __post_init__(self) -> None:
        """Rejects combinations of options that contradict each other."""
        past_offsets_us(self.lidar_config.lidar_horizon_us, self.lidar_config.lidar_interval_us)
        if self.box_detection_config.predict_box_velocity and not self.has_motion_cues:
            raise ValueError(
                "predict_box_velocity needs motion cues: set lidar_horizon_us, or past_camera_offset_us in latent mode.",
            )
        planning = self.planning_config
        path = planning.path_config
        if not planning.predict_waypoints and planning.unicycle_action_config is None:
            raise ValueError(
                "predict_waypoints=false needs unicycle_action_config: without waypoints the action rollout drives.",
            )
        if not planning.flow_config.waypoints_attend_to_actions and path is not None and path.path_attends_to_actions:
            raise ValueError(
                "waypoints_attend_to_actions=false needs path_attends_to_actions=false: the path queries would pass "
                "the actions on to the waypoints.",
            )
        if not planning.flow_config.actions_attend_to_waypoints and path is not None and path.path_attends_to_waypoints:
            raise ValueError(
                "actions_attend_to_waypoints=false needs path_attends_to_waypoints=false: the path queries would pass "
                "the waypoints on to the actions.",
            )
        if path is not None and path.visible_target_point_distances_m is not None:
            if path.path_attends_to_waypoints:
                raise ValueError(
                    "visible_target_point_distances_m needs path_attends_to_waypoints=false: the waypoint queries "
                    "would pass the hidden target points on to the path.",
                )
            if path.path_attends_to_actions:
                raise ValueError(
                    "visible_target_point_distances_m needs path_attends_to_actions=false: the action queries would "
                    "pass the hidden target points on to the path.",
                )
            unknown = set(path.visible_target_point_distances_m) - set(self.required_target_point_distances_m)
            if unknown:
                raise ValueError(f"visible_target_point_distances_m {sorted(unknown)} are not required target points.")
        served = self.planning_conditioning_config.test_time_target_point_distances_m
        if served is not None:
            unknown = set(served) - set(self.required_target_point_distances_m)
            if unknown:
                raise ValueError(
                    f"test_time_target_point_distances_m {sorted(unknown)} are not required target points.",
                )
        probabilities = self.planning_conditioning_config.target_point_dropout_probabilities
        if probabilities is not None:
            if len(probabilities) != len(self.required_target_point_distances_m):
                raise ValueError(
                    f"target_point_dropout_probabilities has {len(probabilities)} values for "
                    f"{len(self.required_target_point_distances_m)} target points.",
                )
            if not all(0.0 <= probability < 1.0 for probability in probabilities):
                raise ValueError(f"target_point_dropout_probabilities {probabilities} must lie in [0, 1).")
            if self.planning_conditioning_config.target_point_dropout_probability:
                raise ValueError(
                    "set target_point_dropout_probability or target_point_dropout_probabilities, not both.",
                )

    @property
    @override
    def required_cameras(self) -> dict[str, list[CameraID]]:
        """The stitched input cameras of each dataset."""
        return self.camera_config.input_cameras

    @property
    @override
    def required_lidars(self) -> dict[str, list[LidarID]]:
        """The input LiDARs of each dataset; none in latent mode."""
        if self.backbone_config.latent:
            return {}
        return INPUT_LIDARS

    @property
    def has_motion_cues(self) -> bool:
        """Whether the input has past LiDAR sweeps, or past camera rows in latent mode."""
        if self.backbone_config.latent:
            return bool(self.camera_config.past_camera_offset_us)
        return bool(past_offsets_us(self.lidar_config.lidar_horizon_us, self.lidar_config.lidar_interval_us))

    @property
    @override
    def required_history_duration_us(self) -> NonNegativeInt:
        """Longest past reach of the LiDAR sweeps, ego history and past camera rows."""
        return max(
            self.lidar_config.lidar_horizon_us,
            self.planning_conditioning_config.ego_history_duration_us,
            self.camera_config.past_camera_offset_us * self.camera_config.num_past_camera_rows,
        )

    @property
    def ego_history_num_steps(self) -> NonNegativeInt:
        """Number of past ego poses fed to the planning decoder; 0 when the history is off."""
        conditioning = self.planning_conditioning_config
        return conditioning.ego_history_duration_us // conditioning.ego_history_interval_us

    @property
    @override
    def required_past_ego_state_interval_us(self) -> PositiveInt | None:
        """Spacing of the past ego poses; None when the ego history is off."""
        if not self.ego_history_num_steps:
            return None
        return self.planning_conditioning_config.ego_history_interval_us

    @property
    @override
    def required_past_camera_interval_us(self) -> PositiveInt | None:
        """Spacing of the past camera rows; None without past rows."""
        return self.camera_config.past_camera_offset_us or None

    @property
    @override
    def required_past_lidar_interval_us(self) -> PositiveInt | None:
        """Spacing of the past LiDAR sweeps; None without past sweeps."""
        if not self.lidar_config.lidar_horizon_us:
            return None
        return self.lidar_config.lidar_interval_us
