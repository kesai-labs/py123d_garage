from __future__ import annotations

from dataclasses import dataclass, field

from py123d.datatypes import CameraID
from typing_extensions import override

from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.config.schema.policy.visualization_config import VisualizationConfig
from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt

# The 2 Hz frame spacing the released model was trained on.
_CONTEXT_FRAME_INTERVAL_US = 500_000


@dataclass
class RectificationConfig:
    """nuScenes front-camera pinhole f-theta renders are rectified to, matching AlpaSim's vavam driver."""

    focal_length_px: float = 1545.0
    principal_point_px: list[float] = field(default_factory=lambda: [960.0, 560.0])
    radial: list[float] = field(default_factory=lambda: [-0.356123, 0.172545, -0.05231, 0.0, 0.0, 0.0])
    tangential: list[float] = field(default_factory=lambda: [-0.00213, 0.000464])
    max_overscan_scale: float = 2.0
    safety_margin_px: float = 10.0


@dataclass
class TrajectoryOptimizerConfig:
    """VaVam-Eco smoothing of the predicted waypoints; fields mirror AlpaSim's driver schema."""

    enabled: bool = False

    smoothness_weight: float = 1.0
    deviation_weight: float = 0.1
    comfort_weight: float = 2.0
    max_iterations: int = 100

    retime_in_frenet: bool = True
    retime_alpha: float = 0.25

    max_deviation: float = 2.0
    max_heading_change: float = 0.5236
    max_speed: float = 15.0
    max_accel: float = 5.0

    max_abs_yaw_rate: float = 0.95
    max_abs_yaw_acc: float = 1.93
    max_lon_acc_pos: float = 4.89
    max_lon_acc_neg: float = -4.05
    max_abs_lon_jerk: float = 8.37


@dataclass
class VavamVisualizationConfig(VisualizationConfig):
    """The shared overlay styles plus the blank BEV sheet the trajectories are drawn on."""

    # Sheet extents in the ego rear-axle frame; 100 m ahead fits 3 s at highway speed.
    bev_min_x_m: float = -10.0
    bev_max_x_m: float = 100.0
    bev_half_width_m: float = 30.0
    bev_pixels_per_meter: float = 6.0


@dataclass
class VavamConfig(AbstractPolicyConfig):
    """VaVAM's config block; the architecture itself comes from the released checkpoint."""

    # -- Config objects --

    # F-theta renders remapped onto the nuScenes pinhole the model was trained on.
    rectification: RectificationConfig = field(default_factory=RectificationConfig)
    # VaVam-Eco smoothing of the predicted waypoints, disabled by default.
    trajectory_optimizer: TrajectoryOptimizerConfig = field(default_factory=TrajectoryOptimizerConfig)
    # Colors and sizes of the rendered view's overlays.
    visualization_config: VavamVisualizationConfig = field(default_factory=VavamVisualizationConfig)

    # -- Values --

    # VaVAM predicts 6 waypoints at 2 Hz.
    trajectory_horizon_us: PositiveInt = 3_000_000
    trajectory_interval_us: PositiveInt = 500_000

    # One target point at the command lookahead; forward turns it into {left, straight, right}.
    required_target_point_distances_m: list[float] = field(default_factory=lambda: [20.0])
    command_lateral_threshold_m: float = 3.0

    input_camera: dict[str, CameraID] = field(
        default_factory=lambda: {
            "nuplan": CameraID.PCAM_F0,
            "carla": CameraID.PCAM_F0,
            "physical-ai-av": CameraID.FTCAM_F0,
            "kesai": CameraID.FTCAM_F0,
        },
    )

    # Temporal context: context_length frames ending at the anchor, spaced at the
    # 2 Hz the model was trained on.
    context_length: int = 1

    @property
    @override
    def required_cameras(self) -> dict[str, list[CameraID]]:
        return {dataset: [camera] for dataset, camera in self.input_camera.items()}

    @property
    @override
    def required_history_duration_us(self) -> NonNegativeInt:
        return (self.context_length - 1) * _CONTEXT_FRAME_INTERVAL_US

    @property
    @override
    def required_past_camera_interval_us(self) -> PositiveInt:
        return _CONTEXT_FRAME_INTERVAL_US
