# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field

from py123d.datatypes import CameraID
from typing_extensions import override

from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig, uniform_offsets_us
from py123d_garage.config.schema.policy.visualization_config import VisualizationConfig
from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt

# The 2 Hz frame spacing the released model was trained on.
_CONTEXT_FRAME_INTERVAL_US = 500_000


@dataclass
class RectificationConfig:
    """nuScenes front-camera pinhole f-theta renders are rectified to, matching AlpaSim's vavam driver."""

    # Focal length of the target pinhole.
    focal_length_px: float = 1545.0
    # Principal point (x, y) of the target pinhole.
    principal_point_px: list[float] = field(default_factory=lambda: [960.0, 560.0])
    # Radial distortion coefficients of the target camera.
    radial: list[float] = field(default_factory=lambda: [-0.356123, 0.172545, -0.05231, 0.0, 0.0, 0.0])
    # Tangential distortion coefficients of the target camera.
    tangential: list[float] = field(default_factory=lambda: [-0.00213, 0.000464])
    # Largest canvas growth allowed to keep the distorted border in view.
    max_overscan_scale: float = 2.0
    # Slack kept around the distorted border.
    safety_margin_px: float = 10.0


@dataclass
class TrajectoryOptimizerConfig:
    """VaVam-Eco smoothing of the predicted waypoints; fields mirror AlpaSim's driver schema."""

    # Smooth the predicted waypoints.
    enabled: bool = False

    # Cost weight of the curvature penalty.
    smoothness_weight: float = 1.0
    # Cost weight of the deviation from the predicted waypoints.
    deviation_weight: float = 0.1
    # Cost weight of the soft comfort penalty.
    comfort_weight: float = 2.0
    # Iteration limit of the optimizer.
    max_iterations: int = 100

    # Retime the waypoints along their path as the optimizer's initial guess.
    retime_in_frenet: bool = True
    # Retiming strength in [0, 1]; higher moves more distance to the front.
    retime_alpha: float = 0.25

    # Largest x and y shift of a waypoint in m.
    max_deviation: float = 2.0
    # Largest heading change of a waypoint in rad.
    max_heading_change: float = 0.5236
    # Speed limit in m/s; the optimizer does not read it.
    max_speed: float = 15.0
    # Acceleration limit in m/s^2; the optimizer does not read it.
    max_accel: float = 5.0

    # Comfort limit on the absolute yaw rate in rad/s.
    max_abs_yaw_rate: float = 0.95
    # Comfort limit on the absolute yaw acceleration in rad/s^2.
    max_abs_yaw_acc: float = 1.93
    # Comfort limit on the longitudinal acceleration in m/s^2.
    max_lon_acc_pos: float = 4.89
    # Comfort limit on the longitudinal deceleration in m/s^2, negative.
    max_lon_acc_neg: float = -4.05
    # Comfort limit on the absolute longitudinal jerk in m/s^3.
    max_abs_lon_jerk: float = 8.37


@dataclass
class VavamVisualizationConfig(VisualizationConfig):
    """The shared overlay styles plus the blank BEV sheet the trajectories are drawn on."""

    # Rear end of the sheet in the ego rear-axle frame.
    bev_min_x_m: float = -10.0
    # Front end of the sheet; 100 m fits 3 s at highway speed.
    bev_max_x_m: float = 100.0
    # Half the lateral width of the sheet.
    bev_half_width_m: float = 30.0
    # Resolution of the sheet.
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

    # Waypoint offsets of the plan: VaVAM predicts 6 waypoints at 2 Hz.
    trajectory_offsets_us: list[PositiveInt] = field(
        default_factory=lambda: uniform_offsets_us(500_000, 3_000_000),
    )

    # One target point at the command lookahead; forward turns it into {left, straight, right}.
    required_target_point_distances_m: list[float] = field(default_factory=lambda: [20.0])
    # Lateral offset of the target point beyond which the command is left or right.
    command_lateral_threshold_m: float = 3.0

    # Front camera the model sees, per dataset.
    input_camera: dict[str, CameraID] = field(
        default_factory=lambda: {
            "nuplan": CameraID.PCAM_F0,
            "carla": CameraID.PCAM_F0,
            "physical-ai-av": CameraID.FTCAM_F0,
            "nurec": CameraID.FTCAM_F0,
        },
    )

    # Number of 2 Hz frames ending at the anchor that the model sees.
    context_length: int = 1

    @property
    @override
    def required_cameras(self) -> dict[str, list[CameraID]]:
        """The input camera of each dataset."""
        return {dataset: [camera] for dataset, camera in self.input_camera.items()}

    @property
    @override
    def required_history_duration_us(self) -> NonNegativeInt:
        """Span of the past context frames."""
        return (self.context_length - 1) * _CONTEXT_FRAME_INTERVAL_US

    @property
    @override
    def required_past_camera_interval_us(self) -> PositiveInt:
        """The 2 Hz spacing of the context frames."""
        return _CONTEXT_FRAME_INTERVAL_US
