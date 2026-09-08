"""Unit tests for LiDAR ground removal: the fixed height cut vs the fitted planes."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from py123d_garage.config.schema.policy.transfuser_config import (
    LidarConfig,
)
from py123d_garage.policy.transfuser.features import _drop_ground_points


def _sloped_road_with_pillars(
    seed: int = 0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    A road plane tilting 3% in x, with two obstacle pillars standing on it.

    Args:
        seed: rng seed

    Returns:
        the point cloud, and the mask marking the pillar points
    """
    rng = np.random.default_rng(seed)
    xs, ys = rng.uniform(-32, 32, 20000), rng.uniform(-32, 32, 20000)
    road = np.stack([xs, ys, 0.03 * xs], axis=1)
    pillars = [
        np.stack(
            [
                center_x + rng.uniform(-0.5, 0.5, 500),
                center_y + rng.uniform(-0.5, 0.5, 500),
                0.03 * center_x + rng.uniform(0.3, 2.0, 500),
            ],
            axis=1,
        )
        for center_x, center_y in [(10.0, 4.0), (-8.0, -6.0)]
    ]
    cloud = np.concatenate([road, *pillars], axis=0).astype(np.float32)
    is_pillar = np.zeros(len(cloud), dtype=bool)
    is_pillar[len(road) :] = True
    return cloud, is_pillar


def _kept_fractions(cloud, is_pillar, kept) -> tuple[float, float]:
    """Fraction of pillar points and of road points that survived removal."""
    keys = {tuple(row) for row in kept}
    pillar_kept = sum(tuple(row) in keys for row in cloud[is_pillar]) / is_pillar.sum()
    road_kept = sum(tuple(row) in keys for row in cloud[~is_pillar]) / (~is_pillar).sum()
    return pillar_kept, road_kept


def test_unknown_method_raises() -> None:
    cloud, _ = _sloped_road_with_pillars()
    with pytest.raises(ValueError, match="Unknown ground_removal"):
        _drop_ground_points(
            cloud,
            LidarConfig(ground_removal="bogus"),
        )


def test_fixed_z_leaks_a_sloped_road() -> None:
    """The failure mode the fitted planes fix: one height cut cannot follow a slope."""
    cloud, is_pillar = _sloped_road_with_pillars()
    kept = _drop_ground_points(
        cloud,
        LidarConfig(ground_removal="fixed_z"),
    )
    _, road_kept = _kept_fractions(cloud, is_pillar, kept)
    assert road_kept > 0.2


@pytest.mark.slow
def test_fitted_plane_drops_the_road_and_keeps_the_obstacles() -> None:
    """Marked slow: the first call pays numba's compile."""
    cloud, is_pillar = _sloped_road_with_pillars()
    kept = _drop_ground_points(
        cloud,
        LidarConfig(ground_removal="fitted_plane"),
    )
    pillar_kept, road_kept = _kept_fractions(cloud, is_pillar, kept)
    assert pillar_kept > 0.9
    assert road_kept < 0.01
