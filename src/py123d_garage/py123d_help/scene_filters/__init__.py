# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Declares which scenes a run sees, and the predicates that decide it."""

from __future__ import annotations

from py123d_garage.py123d_help.scene_filters.filter_fns import (
    has_box_detections,
    is_navtest_scene,
    is_not_nurec_clip,
    is_not_reversing_log,
    is_nurec_hard_scene,
    num_navtest_scenes,
    plausible_ego_motion,
    plausible_route,
)
from py123d_garage.py123d_help.scene_filters.scene_filter import GarageSceneFilter

__all__ = [
    "GarageSceneFilter",
    "has_box_detections",
    "is_navtest_scene",
    "is_not_nurec_clip",
    "is_not_reversing_log",
    "is_nurec_hard_scene",
    "num_navtest_scenes",
    "plausible_ego_motion",
    "plausible_route",
]
