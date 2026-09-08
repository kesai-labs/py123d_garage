"""Unit tests for the route polyline the CARLA agent conditions a policy on."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from py123d_garage.evaluation.carla.help.route import ROUTE_RESOLUTION_M, Route


class _Location:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y, self.z = x, y, 0.0


class _Transform:
    def __init__(self, x: float, y: float) -> None:
        self.location = _Location(x, y)


def _right_angle_route() -> list[tuple[Any, Any]]:
    """100 m east then 100 m to the CARLA right, at the leaderboard's 1 m hop resolution."""
    east = [(_Transform(float(step), 0.0), None) for step in range(101)]
    south = [(_Transform(100.0, float(step)), None) for step in range(101)]
    return east + south


def test_polyline_is_uniformly_resampled_and_mirrored_into_iso() -> None:
    metadata = Route(_right_angle_route(), extension_m=0.0).metadata
    assert metadata.resolution_m == ROUTE_RESOLUTION_M
    assert metadata.total_arc_m == pytest.approx(200.0)
    assert len(metadata.polyline_x) == 2001
    # CARLA's right turn is -y once mirrored into the ISO frame.
    assert metadata.polyline_xyz[1500][:2] == pytest.approx([100.0, -50.0])


def test_extension_runs_past_the_destination_along_the_final_heading() -> None:
    metadata = Route(_right_angle_route(), extension_m=45.0).metadata
    assert metadata.total_arc_m == pytest.approx(245.0)
    assert metadata.polyline_xyz[2200][:2] == pytest.approx([100.0, -120.0])


def test_progress_tracks_arc_length_along_the_whole_route() -> None:
    route = Route(_right_angle_route(), extension_m=45.0)
    driven = np.concatenate(
        [
            np.stack([np.arange(0.0, 100.0), np.zeros(100)], axis=1),
            np.stack([np.full(100, 100.0), -np.arange(0.0, 100.0)], axis=1),
        ],
    )
    for step, position in enumerate(driven):
        assert route.advance(position) == pytest.approx(float(step), abs=ROUTE_RESOLUTION_M)


def test_progress_never_rewinds_when_the_ego_is_pushed_back() -> None:
    route = Route(_right_angle_route(), extension_m=0.0)
    for position in np.stack([np.arange(0.0, 50.0), np.zeros(50)], axis=1):
        route.advance(position)
    reached = route.advance(np.array([49.0, 0.0]))
    assert route.advance(np.array([40.0, 0.0])) == pytest.approx(reached)
