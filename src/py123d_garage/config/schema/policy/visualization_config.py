# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryStyleConfig:
    """Drawing style of one trajectory overlay."""

    # Line and pose color as 0-255 RGB.
    color_rgb: list[int] = field(default_factory=lambda: [0, 0, 0])
    # Thickness of the line between poses.
    line_thickness_pixel: int = 2
    # Radius of the dot drawn at each pose.
    pose_radius_pixel: int = 3


@dataclass
class VisualizationConfig:
    """Overlay styles shared by every policy's views; a policy subclasses it for its own options."""

    # Style of the predicted trajectory, tab10 orange like the plan in the AlpaSim view.
    prediction_trajectory_style: TrajectoryStyleConfig = field(
        default_factory=lambda: TrajectoryStyleConfig(color_rgb=[255, 127, 14], pose_radius_pixel=1),
    )
    # Style of the ground-truth trajectory, tab10 green.
    ground_truth_trajectory_style: TrajectoryStyleConfig = field(
        default_factory=lambda: TrajectoryStyleConfig(color_rgb=[44, 160, 44], pose_radius_pixel=1),
    )
    # Color of the target points, drawn as circles with a cross.
    target_point_color_rgb: list[int] = field(default_factory=lambda: [31, 119, 180])
    # Radius of the target point circles.
    target_point_radius_pixel: int = 4
    # Color of the ego box outline.
    ego_box_color_rgb: list[int] = field(default_factory=lambda: [148, 103, 189])
    # Line thickness of the ego box outline.
    ego_box_thickness_pixel: int = 2
