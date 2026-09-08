"""Unit tests for the CARLA localization math: compass preprocessing and inverse GPS projections."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from py123d_garage.evaluation.carla.help.localization import (
    find_gps_ref,
    gnss_to_carla,
    preprocess_compass,
    route_gps_to_carla,
)


def test_preprocess_compass_shifts_north_to_carla_yaw() -> None:
    assert preprocess_compass(math.pi / 2.0) == pytest.approx(0.0, abs=1e-12)
    assert preprocess_compass(math.pi) == pytest.approx(math.pi / 2.0)
    assert preprocess_compass(0.0) == pytest.approx(-math.pi / 2.0)


def test_preprocess_compass_nan_reads_as_zero() -> None:
    assert preprocess_compass(float("nan")) == pytest.approx(-math.pi / 2.0)


def test_route_gps_reference_maps_to_origin() -> None:
    out = route_gps_to_carla(np.array([0.0, 0.0, 5.0]), 0.0, 0.0)
    assert np.allclose(out, [0.0, 0.0, 5.0])


def test_route_gps_one_millidegree_east_is_positive_x() -> None:
    # One degree of longitude at the equator spans 111319.49 m.
    out = route_gps_to_carla(np.array([0.0, 0.001, 5.0]), 0.0, 0.0)
    assert np.allclose(out, [111.3194908, 0.0, 5.0], atol=1e-3)


def test_route_gps_one_millidegree_north_is_negative_y() -> None:
    out = route_gps_to_carla(np.array([0.001, 0.0, 0.0]), 0.0, 0.0)
    assert np.allclose(out, [0.0, -111.3194908, 0.0], atol=1e-3)


def test_route_gps_scale_shrinks_with_reference_latitude() -> None:
    # cos(60 deg) = 0.5 halves the meters per degree of longitude.
    out = route_gps_to_carla(np.array([60.0, 8.001, 0.0]), 60.0, 8.0)
    assert np.allclose(out, [55.6597454, 0.0, 0.0], atol=1e-3)


def test_gnss_reference_maps_to_origin() -> None:
    out = gnss_to_carla(np.array([49.0, 8.0, 1.5]), 49.0, 8.0)
    assert np.allclose(out, [0.0, 0.0, 1.5])


def test_gnss_meridian_arc_north_is_positive_y() -> None:
    # 0.001 deg of spherical meridian arc, scaled by the 0.9996 tmerc factor.
    out = gnss_to_carla(np.array([49.001, 8.0, 3.0]), 49.0, 8.0, carla_version="0.9.16")
    assert np.allclose(out, [0.0, 111.2749630, 3.0], atol=1e-3)


def test_gnss_equator_east_is_positive_x() -> None:
    out = gnss_to_carla(np.array([0.0, 8.01, 0.0]), 0.0, 8.0, carla_version="0.9.16")
    assert np.allclose(out, [1112.749630, 0.0, 0.0], atol=1e-2)


def test_gnss_0915_matches_the_route_projection() -> None:
    gnss = np.array([49.001, 8.002, 3.0])
    out = gnss_to_carla(gnss, 49.0, 8.0, carla_version="0.9.15")
    np.testing.assert_allclose(out, route_gps_to_carla(gnss, 49.0, 8.0))


def test_find_gps_ref_recovers_the_map_reference() -> None:
    lat_ref, lon_ref = 49.0, 8.0
    gps = {"lat": 49.002, "lon": 8.003, "z": 0.0}
    world = route_gps_to_carla(
        np.array([gps["lat"], gps["lon"], gps["z"]]),
        lat_ref,
        lon_ref,
    )
    transform = SimpleNamespace(
        location=SimpleNamespace(x=float(world[0]), y=float(world[1])),
    )
    solved_lat, solved_lon = find_gps_ref(
        [(gps, None)],
        cast("list[tuple[Any, object]]", [(transform, None)]),
    )
    assert solved_lat == pytest.approx(lat_ref, abs=1e-8)
    assert solved_lon == pytest.approx(lon_ref, abs=1e-8)
