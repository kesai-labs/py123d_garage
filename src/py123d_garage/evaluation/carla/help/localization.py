from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from py123d.geometry.utils.rotation_utils import normalize_angle

if TYPE_CHECKING:
    import carla

EARTH_RADIUS_EQUA = 6378137.0  # Constant of the CARLA leaderboard GPS projection.
TMERC_SCALE = 0.9996  # Scale factor hardcoded in CARLA 0.9.16's GeoLocation::Transform.


def preprocess_compass(compass: float) -> float:
    """
    Turns an IMU compass reading into a CARLA yaw in [-pi, pi].

    The compass points north (90 degrees off CARLA's x-axis) and is NaN on the
    first simulation ticks.

    Args:
        compass: compass reading in radians.

    Returns:
        ego yaw in radians in the CARLA frame.
    """
    if math.isnan(compass):
        compass = 0.0
    return float(normalize_angle(compass - math.radians(90.0)))


def route_gps_to_carla(
    gps: npt.NDArray[np.float64],
    lat_ref: float,
    lon_ref: float,
) -> npt.NDArray[np.float64]:
    """
    Inverts the equatorial Mercator projection the leaderboard uses for the route plan.

    Args:
        gps: (lat, lon, z) reading.
        lat_ref: latitude of the map's geo-reference.
        lon_ref: longitude of the map's geo-reference.

    Returns:
        (x, y, z) CARLA world coordinates in meters.
    """
    lat, lon, z = gps
    scale = math.cos(math.radians(lat_ref))
    my = math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
    mx = lon * math.pi * EARTH_RADIUS_EQUA * scale / 180.0
    y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - my
    x = mx - scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    return np.array([x, y, z])


def gnss_to_carla(
    gnss: npt.NDArray[np.float64],
    lat_ref: float,
    lon_ref: float,
    carla_version: str = "0.9.15",
) -> npt.NDArray[np.float64]:
    """
    Inverts the GNSS sensor projection of the given CARLA version.

    CARLA 0.9.16 projects the sensor through a spherical transverse Mercator;
    0.9.15 and earlier use the same equatorial Mercator as the route plan.

    Args:
        gnss: (lat, lon, z) reading of the GNSS sensor.
        lat_ref: latitude of the map's geo-reference.
        lon_ref: longitude of the map's geo-reference.
        carla_version: version of the simulator the reading comes from.

    Returns:
        (x, y, z) CARLA world coordinates in meters.
    """
    if carla_version < "0.9.16":
        return route_gps_to_carla(gnss, lat_ref, lon_ref)
    lat, lon, z = gnss
    phi = math.radians(lat)
    delta_lambda = math.radians(lon - lon_ref)
    b = math.cos(phi) * math.sin(delta_lambda)
    x = 0.5 * TMERC_SCALE * EARTH_RADIUS_EQUA * math.log((1.0 + b) / (1.0 - b))
    y = TMERC_SCALE * EARTH_RADIUS_EQUA * (math.atan(math.tan(phi) / math.cos(delta_lambda)) - math.radians(lat_ref))
    return np.array([x, y, z])


def find_gps_ref(
    global_plan_gps: list[tuple[dict[str, float], object]],
    global_plan_world_coord: list[tuple[carla.Transform, object]],
) -> tuple[float, float]:
    """
    Solves the map's geo-reference, which the leaderboard does not expose, from the route plan.

    The plan comes in both GPS and world coordinates; the reference is the fixed
    point of the leaderboard's Mercator projection relating the two.

    Args:
        global_plan_gps: route plan as ({'lat', 'lon', 'z'}, road option) tuples.
        global_plan_world_coord: the same plan in CARLA world coordinates.

    Returns:
        (lat_ref, lon_ref) of the map.
    """
    location = global_plan_world_coord[0][0].location
    lat, lon = global_plan_gps[0][0]["lat"], global_plan_gps[0][0]["lon"]

    def mercator_ordinate(lat_deg: float) -> float:
        return math.log(math.tan((90.0 + lat_deg) * math.pi / 360.0))

    lat_ref = lat
    for _ in range(20):
        ordinate = mercator_ordinate(lat) + location.y / (EARTH_RADIUS_EQUA * math.cos(math.radians(lat_ref)))
        lat_ref = 360.0 * math.atan(math.exp(ordinate)) / math.pi - 90.0
    lon_ref = lon - location.x * 180.0 / (math.pi * EARTH_RADIUS_EQUA * math.cos(math.radians(lat_ref)))
    return lat_ref, lon_ref
