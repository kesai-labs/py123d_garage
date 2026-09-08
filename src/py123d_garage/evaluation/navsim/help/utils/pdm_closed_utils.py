# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import shapely.geometry as geom
from py123d.api import MapAPI
from py123d.datatypes import BaseMapSurfaceObject, EgoStateSE2, Lane, LaneGroup, MapLayer
from py123d.geometry import OccupancyMap2D, PolylineSE2, PoseSE2Index, Vector2D
from py123d.geometry.transform.transform_se2 import translate_se2_array_along_body_frame
from py123d.geometry.utils.rotation_utils import normalize_angle
from shapely.geometry import Point

from py123d_garage.evaluation.navsim.help.utils.graph_search.dijkstra import Dijkstra
from py123d_garage.evaluation.navsim.help.utils.route_utils import route_lane_group_correction


def build_route_dicts(
    map_api: MapAPI,
    route_roadblock_ids: list[int],
) -> tuple[dict[int, LaneGroup], dict[int, Lane]]:
    """
    Builds roadblock and lane dictionaries of the target route from the map-api.

    Args:
        map_api: map interface
        route_roadblock_ids: ID's of on-route roadblocks

    Returns:
        tuple of (route_roadblock_dict, route_lane_dict)
    """
    route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

    route_lane_group_dict: dict[int, LaneGroup] = {}
    route_lane_dict: dict[int, Lane] = {}

    for id_ in route_roadblock_ids:
        _lane_group = map_api.get_map_object_in_layer(id_, MapLayer.LANE_GROUP)

        if isinstance(_lane_group, LaneGroup):
            route_lane_group_dict[int(_lane_group.object_id)] = _lane_group

            for lane in _lane_group.lanes:
                route_lane_dict[int(lane.object_id)] = lane

    return route_lane_group_dict, route_lane_dict


def correct_route_lane_groups(
    ego_state_se2: EgoStateSE2,
    map_api: MapAPI,
    route_lane_group_dict: dict[int, LaneGroup],
) -> tuple[dict[int, LaneGroup], dict[int, Lane]]:
    """
    Corrects the roadblock route and rebuilds lane-graph dictionaries.

    Args:
        ego_state_se2: state of the ego vehicle
        map_api: map interface
        route_lane_group_dict: current lane group dict (used for correction context)

    Returns:
        tuple of (corrected route_roadblock_dict, corrected route_lane_dict)
    """
    _, corrected_ids = route_lane_group_correction(
        ego_pose_se2=ego_state_se2.rear_axle_se2,
        map_api=map_api,
        route_lane_group_dict=route_lane_group_dict,
    )
    return build_route_dicts(map_api, corrected_ids)


def build_drivable_area_occupancy_map(
    map_api: MapAPI,
    ego_state_se2: EgoStateSE2,
    map_radius: float = 50.0,
    layers: list[MapLayer] | None = None,
) -> OccupancyMap2D:
    """
    Builds an occupancy map of drivable surfaces around ego from the given map layers.

    Args:
        map_api: map interface
        ego_state_se2: state of the ego vehicle
        map_radius: radius around ego to query [m], defaults to 50.0
        layers: map layers treated as drivable

    Returns:
        occupancy map of drivable surface polygons
    """
    if layers is None:
        layers = [
            MapLayer.LANE,
            MapLayer.LANE_GROUP,
            MapLayer.INTERSECTION,
            MapLayer.GENERIC_DRIVABLE,
            MapLayer.CARPARK,
        ]
    query_dict = map_api.get_map_objects_in_radius(
        point=ego_state_se2.center_2d,
        radius=map_radius,
        layers=layers,  # type: ignore[arg-type]
    )
    drivable_objects_dict: dict[str, geom.Polygon] = {}
    for map_layer in query_dict:
        for map_object in query_dict[map_layer]:
            assert isinstance(map_object, BaseMapSurfaceObject), f"Expected Surface, got {type(map_object)}"
            drivable_objects_dict[f"{map_layer.serialize()}-{map_object.object_id}"] = map_object.shapely_polygon

    return OccupancyMap2D.from_dict(drivable_objects_dict)  # type: ignore[return-value]


def _get_intersecting_lanes(
    ego_state_se2: EgoStateSE2,
    route_lane_dict: dict[int, Lane],
    drivable_area_map: OccupancyMap2D,
) -> tuple[list[Lane], list[float]]:
    """
    Returns on-route lanes and heading errors where ego-vehicle intersects.

    Args:
        ego_state_se2: state of ego-vehicle
        route_lane_dict: on-route lane dictionary
        drivable_area_map: drivable area occupancy map

    Returns:
        tuple of lists with lane objects and heading errors [rad].
    """
    ego_se2_array: npt.NDArray[np.float64] = ego_state_se2.rear_axle_se2.array
    ego_rear_axle_point: Point = Point(*ego_se2_array[PoseSE2Index.XY])

    intersecting_lanes = drivable_area_map.intersects(ego_rear_axle_point)

    on_route_lanes: list[Lane] = []
    on_route_heading_errors: list[float] = []
    for lane_token in intersecting_lanes:
        assert isinstance(lane_token, str), f"Expected lane_id of type int, got {type(lane_token)}"
        map_layer, lane_id = lane_token.split("-")
        lane_id = int(lane_id)
        map_layer = MapLayer.from_arbitrary(map_layer)
        if map_layer == MapLayer.LANE and lane_id in route_lane_dict:
            lane_object = route_lane_dict[lane_id]
            lane_centerline_se2: PolylineSE2 = lane_object.centerline.polyline_se2
            lane_centerline_se2_array = lane_centerline_se2.array

            lane_distances = (
                ego_se2_array[None, ..., PoseSE2Index.XY] - lane_centerline_se2_array[..., PoseSE2Index.XY]
            ) ** 2
            lane_distances = lane_distances.sum(axis=-1) ** 0.5

            heading_error = (
                lane_centerline_se2[np.argmin(lane_distances)][
                    ...,
                    PoseSE2Index.YAW,
                ]
                - ego_se2_array[PoseSE2Index.YAW]
            )
            on_route_lanes.append(lane_object)
            on_route_heading_errors.append(
                float(np.abs(normalize_angle(heading_error))),
            )

    return on_route_lanes, on_route_heading_errors


def get_starting_lane(
    ego_state_se2: EgoStateSE2,
    route_lane_dict: dict[int, Lane],
    drivable_area_map: OccupancyMap2D,
) -> Lane:
    """
    Returns the most suitable starting lane in ego's vicinity.

    Args:
        ego_state_se2: state of ego-vehicle
        route_lane_dict: on-route lane dictionary
        drivable_area_map: drivable area occupancy map

    Returns:
        lane object (on-route)
    """
    on_route_lanes, heading_error = _get_intersecting_lanes(
        ego_state_se2,
        route_lane_dict,
        drivable_area_map,
    )

    if len(on_route_lanes) > 0:
        # 1. Option: find lanes from lane occupancy-map; select lane with lowest heading error
        return on_route_lanes[np.argmin(np.abs(heading_error))]

    # 2. Option: find any intersecting or close lane on-route
    starting_lane: Lane | None = None
    closest_distance = np.inf
    for lane in route_lane_dict.values():
        if lane.shapely_polygon.contains(ego_state_se2.center_2d.shapely_point):
            return lane

        distance = lane.shapely_polygon.distance(
            ego_state_se2.bounding_box_se2.shapely_polygon,
        )
        if distance < closest_distance:
            starting_lane = lane
            closest_distance = distance

    assert starting_lane is not None, "No starting lane found in route lane dict. Check map and route consistency."
    return starting_lane


def get_centerline_as_polyline_se2(
    current_lane: Lane,
    route_lane_group_dict: dict[int, LaneGroup],
    route_lane_dict: dict[int, Lane],
    ego_state_se2: EgoStateSE2 | None = None,
    search_depth: int = 30,
    max_centerline_start_offset: float = 200.0,
    duplicate_pose_eps: float = 1e-3,
) -> PolylineSE2:
    """
    Applies a Dijkstra search on the lane-graph to retrieve the discrete centerline.

    Args:
        current_lane: lane object of starting lane.
        route_lane_group_dict: on-route lane group dictionary
        route_lane_dict: on-route lane dictionary
        ego_state_se2: ego state used to sanity-check the centerline starts near ego
        search_depth: depth of search (for runtime), defaults to 30
        max_centerline_start_offset: [m] max allowed gap between first centerline point and ego
        duplicate_pose_eps: [m] consecutive poses closer than this are dropped (stitch boundaries / zero-length connectors)

    Returns:
        list of discrete states on centerline (x,y,θ)
    """
    lane_groups = list(route_lane_group_dict.values())
    lane_group_ids = list(route_lane_group_dict.keys())

    assert current_lane.lane_group_id in lane_group_ids, (
        f"Starting lane {current_lane.object_id} has lane_group_id {current_lane.lane_group_id} "
        f"not in route lane groups {lane_group_ids}."
    )
    start_idx = int(
        np.argmax(np.array(lane_group_ids) == current_lane.lane_group_id),
    )
    lane_group_window = lane_groups[start_idx : start_idx + search_depth]
    assert len(lane_group_window) > 0, "Empty lane_group_window — start_idx is past route end."

    graph_search = Dijkstra(current_lane, list(route_lane_dict.keys()))
    route_plan, path_found = graph_search.search(lane_group_window[-1])

    centerline_sublines: list[npt.NDArray[np.float64]] = [lane.centerline.polyline_se2.array for lane in route_plan]
    stacked = np.vstack(centerline_sublines)

    assert np.isfinite(stacked).all(), (
        f"Non-finite values in stitched centerline (start_lane={current_lane.object_id}, "
        f"route_plan_len={len(route_plan)}, path_found={path_found})."
    )

    # TODO: @DanielDauner: Refactor these checks and assertions.
    # Drop consecutive near-duplicate poses. Lane-to-lane stitch boundaries always
    # duplicate one point; zero-length lane connectors (common in Boston/Singapore
    # intersections) produce additional duplicates that make shapely's project()
    # return NaN and that feed singular rows into the LQR fit.
    if stacked.shape[0] > 1:
        gaps = np.linalg.norm(np.diff(stacked[:, :2], axis=0), axis=1)
        keep = np.concatenate([[True], gaps > duplicate_pose_eps])
        stacked = stacked[keep]

    assert stacked.shape[0] >= 2, (
        f"Centerline collapsed to {stacked.shape[0]} pose(s) after dedup "
        f"(start_lane={current_lane.object_id}, route_plan_len={len(route_plan)})."
    )

    if ego_state_se2 is not None:
        ego_xy = ego_state_se2.rear_axle_se2.array[:2]
        start_offset = float(
            np.linalg.norm(stacked[:, :2] - ego_xy, axis=1).min(),
        )
        assert start_offset < max_centerline_start_offset, (
            f"Centerline starts {start_offset:.1f}m from ego "
            f"(start_lane={current_lane.object_id}, route_plan_len={len(route_plan)}, "
            f"path_found={path_found}). Route correction or Dijkstra fallback likely picked a remote lane."
        )

    return PolylineSE2.from_array(stacked)


def get_proposal_paths(
    centerline_polyline_se2: PolylineSE2,
    lateral_offsets: Sequence[float] | None,
) -> list[PolylineSE2]:
    """
    Builds proposal paths: centerline at index 0, plus optional lateral offsets.

    Args:
        centerline_polyline_se2: the centerline path to offset from
        lateral_offsets: optional centerline offsets for proposals

    Returns:
        list of paths (index 0 is centerline)
    """

    output_paths: list[PolylineSE2] = [centerline_polyline_se2]
    if lateral_offsets is not None:
        for lateral_offset in lateral_offsets:
            lateral_se2_array = translate_se2_array_along_body_frame(
                centerline_polyline_se2.array,
                Vector2D(0.0, lateral_offset),
            )
            output_paths.append(PolylineSE2.from_array(lateral_se2_array))
    return output_paths
