"""Glue between the garage and py123d's scene API."""

from __future__ import annotations

# Registers the label classes of lead's CARLA export.
import py123d_garage.py123d_help.misc.carla_labels
from py123d_garage.py123d_help.misc import FRONT_CAMERAS, INPUT_LIDARS
from py123d_garage.py123d_help.scene_builders import (
    VerboseProcessPoolExecutor,
    VerboseThreadPoolExecutor,
    build_scene_builder,
    find_log_names,
)
from py123d_garage.py123d_help.scene_filters import (
    GarageSceneFilter,
    has_box_detections,
    is_navtest_scene,
    plausible_ego_motion,
    plausible_route,
)
from py123d_garage.py123d_help.scene_readers import (
    InsufficientRouteError,
    get_target_points,
)

__all__ = [
    "FRONT_CAMERAS",
    "INPUT_LIDARS",
    "GarageSceneFilter",
    "InsufficientRouteError",
    "VerboseProcessPoolExecutor",
    "VerboseThreadPoolExecutor",
    "build_scene_builder",
    "find_log_names",
    "get_target_points",
    "has_box_detections",
    "is_navtest_scene",
    "plausible_ego_motion",
    "plausible_route",
]
