from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryStyleConfig:
    color_rgb: list[int] = field(default_factory=lambda: [0, 0, 0])
    line_thickness_pixel: int = 2
    pose_radius_pixel: int = 3


@dataclass
class VisualizationConfig:
    """Overlay styles shared by every policy's views; a policy subclasses it for its own options."""

    # Colors are matplotlib's tab10: red, green, blue, purple.
    prediction_trajectory_style: TrajectoryStyleConfig = field(
        default_factory=lambda: TrajectoryStyleConfig(color_rgb=[214, 39, 40]),
    )
    ground_truth_trajectory_style: TrajectoryStyleConfig = field(
        default_factory=lambda: TrajectoryStyleConfig(color_rgb=[44, 160, 44]),
    )
    # Target points are filled circles.
    target_point_color_rgb: list[int] = field(default_factory=lambda: [31, 119, 180])
    target_point_radius_pixel: int = 6
    # The ego box outline.
    ego_box_color_rgb: list[int] = field(default_factory=lambda: [148, 103, 189])
    ego_box_thickness_pixel: int = 2
