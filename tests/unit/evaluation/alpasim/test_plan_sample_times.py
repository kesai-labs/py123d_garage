# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The served poses sit on the plan's own 10 Hz clock, so no pose moves between drive calls."""

from __future__ import annotations

import math

import numpy as np

from py123d_garage.evaluation.alpasim.help.plan import (
    SERVE_HORIZON_US,
    SERVE_INTERVAL_US,
    CachedPlan,
    make_plan,
    sample_plan,
)

_CREATED_US = 1_000_000
_EGO_POSE_LOCAL = np.array([10.0, -5.0, 0.3, math.pi / 2])
_OFFSETS_US = [500_000 * step for step in range(1, 9)]


def _plan(offsets_us: list[int] | None = None) -> CachedPlan:
    offsets_us = _OFFSETS_US if offsets_us is None else offsets_us
    poses_rig = np.zeros((len(offsets_us), 3))
    poses_rig[:, 0] = 2.0 * np.arange(1, len(offsets_us) + 1)
    return make_plan(poses_rig, _EGO_POSE_LOCAL, _CREATED_US, offsets_us)


def test_make_plan_anchors_the_current_pose_and_maps_rig_poses_into_the_local_frame() -> None:
    plan = _plan()
    assert plan.created_time_us == _CREATED_US
    np.testing.assert_allclose(plan.times_s, [0.0, *[offset / 1e6 for offset in _OFFSETS_US]])
    np.testing.assert_allclose(plan.positions_xy[0], _EGO_POSE_LOCAL[:2])
    np.testing.assert_allclose(plan.positions_xy[1:, 0], 10.0, atol=1e-12)
    np.testing.assert_allclose(plan.positions_xy[1:, 1], -5.0 + 2.0 * np.arange(1, 9))
    np.testing.assert_allclose(plan.yaws, math.pi / 2)


def test_samples_start_now_on_the_plans_clock_and_end_with_the_plan() -> None:
    samples = sample_plan(_plan(), _CREATED_US)
    assert samples is not None
    np.testing.assert_array_equal(samples.timestamps_us, _CREATED_US + SERVE_INTERVAL_US * np.arange(41))
    np.testing.assert_allclose(samples.positions_xy[1], [10.0, -4.6])
    np.testing.assert_allclose(samples.positions_xy[-1], [10.0, 11.0])
    np.testing.assert_allclose(samples.yaws, math.pi / 2)


def test_a_later_call_serves_the_same_poses_minus_the_elapsed_ones() -> None:
    plan = _plan()
    first = sample_plan(plan, _CREATED_US)
    later = sample_plan(plan, _CREATED_US + 250_000)
    assert first is not None
    assert later is not None
    assert later.timestamps_us[0] == _CREATED_US + 300_000
    np.testing.assert_array_equal(later.timestamps_us, first.timestamps_us[3:])
    np.testing.assert_array_equal(later.positions_xy, first.positions_xy[3:])
    np.testing.assert_array_equal(later.yaws, first.yaws[3:])


def test_the_horizon_caps_at_five_seconds() -> None:
    plan = _plan([500_000 * step for step in range(1, 17)])
    samples = sample_plan(plan, _CREATED_US + 1_000_000)
    assert samples is not None
    assert len(samples.timestamps_us) == SERVE_HORIZON_US // SERVE_INTERVAL_US + 1
    assert samples.timestamps_us[0] == _CREATED_US + 1_000_000
    assert samples.timestamps_us[-1] == _CREATED_US + 6_000_000


def test_a_spent_plan_yields_nothing() -> None:
    plan = _plan()
    assert sample_plan(plan, _CREATED_US + 3_900_000) is not None
    assert sample_plan(plan, _CREATED_US + 3_950_000) is None
    assert sample_plan(plan, _CREATED_US + 4_000_000) is None


def test_a_request_before_the_plan_was_made_clamps_to_its_start() -> None:
    plan = _plan()
    early = sample_plan(plan, _CREATED_US - 700_000)
    now = sample_plan(plan, _CREATED_US)
    assert early is not None
    assert now is not None
    np.testing.assert_array_equal(early.timestamps_us, now.timestamps_us)
