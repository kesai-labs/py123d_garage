# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from typing import cast

import networkx as nx
import numpy as np
import shapely.geometry as geom
from py123d.api import MapAPI, SceneAPI
from py123d.datatypes import Intersection, LaneGroup, MapLayer
from py123d.geometry import OccupancyMap2D, Point2D, PoseSE2, PoseSE2Index
from py123d.geometry.utils.rotation_utils import normalize_angle

from py123d_garage.evaluation.navsim.help.utils.graph_search.bfs_roadblock import BreadthFirstSearchLaneGroup

_CANDIDATE_FALLBACK_RADII: list[float] = [5.0, 20.0, 50.0]
_USE_NUPLAN_DEFAULT_ROUTE: bool = True
_TARGET_ROUTE_LOOKAHEAD_TIME_S: float = 60.0


def get_route_lane_group_ids(scene_api: SceneAPI) -> list[int]:
    """
    Returns the on-route lane group ids for the scene.

    Uses nuPlan's logged route roadblocks when available, otherwise infers the route via
    :func:`_infer_route_lane_group_ids`.

    Args:
        scene_api: scene interface providing map and ego state access

    Returns:
        ordered on-route lane group ids (empty if none can be determined)
    """

    route_lane_group_ids: list[int] = []
    if scene_api.get_map_metadata() is not None:
        dataset_name = scene_api.get_log_metadata().dataset
        if (
            "nuplan" in dataset_name
            and _USE_NUPLAN_DEFAULT_ROUTE
            and ("scenario" in scene_api.get_all_custom_modality_metadatas())
        ):
            modality = scene_api.get_custom_modality_at_iteration(
                0,
                "scenario",
            )
            assert modality is not None
            route_lane_group_ids = [int(id_) for id_ in modality.data["route_roadblock_ids"]]

        if len(route_lane_group_ids) == 0:
            route_lane_group_ids = _infer_route_lane_group_ids(scene_api)
    return route_lane_group_ids


def _infer_route_lane_group_ids(scene_api: SceneAPI) -> list[int]:
    """
    Infers the route lane group ids via shortest-path search on the lane-group graph.

    Looks up the ego position ``_TARGET_ROUTE_LOOKAHEAD_TIME_S`` ahead (via an oracle), then finds
    the shortest lane-group path from the current position's candidates to the look-ahead candidates.

    Args:
        scene_api: Arrow-backed scene interface

    Returns:
        lane group ids along the inferred route (empty if no path is found)
    """

    initial_ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
    assert initial_ego_state_se3 is not None, "Ego state modality not found at iteration 0."

    target_ts_us = initial_ego_state_se3.timestamp.time_us + int(
        _TARGET_ROUTE_LOOKAHEAD_TIME_S * 1e6,
    )
    end_ego_state_se3 = scene_api.get_ego_state_se3_at_timestamp(
        target_ts_us,
        criteria="nearest",
    )
    map_api = scene_api.get_map_api()
    assert end_ego_state_se3 is not None, (
        f"Ego state modality not found at iteration {scene_api.number_of_iterations - 1}."
    )
    assert map_api is not None, "Map API not found in Agent API."

    # Query nearest lane groups to initial and end ego states, several candidates for start and end lane groups.
    start_candidates = _query_lane_group_candidates(
        map_api,
        initial_ego_state_se3.center_2d,
    )
    end_candidates = _query_lane_group_candidates(
        map_api,
        end_ego_state_se3.center_2d,
    )
    if not start_candidates or not end_candidates:
        return []

    # Use lane group digraph to find optional lane group sequence from start to end candidates.
    # py123d annotates the return as a bare nx.DiGraph; the nodes are the layer's object ids.
    lane_group_digraph = cast(
        "nx.DiGraph[int]",
        map_api.get_layer_graph(  # pyright: ignore[reportUnknownMemberType]
            layer="lane_group",
        ),
    )

    best_path: list[int] = []
    for start_id in start_candidates:
        if start_id not in lane_group_digraph:
            continue
        for end_id in end_candidates:
            if end_id not in lane_group_digraph:
                continue
            try:
                path: list[int] = nx.shortest_path(  # pyright: ignore[reportUnknownMemberType]
                    lane_group_digraph,
                    source=start_id,
                    target=end_id,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if not best_path or len(path) < len(best_path):
                best_path = [int(node) for node in path]

    return best_path


def _query_lane_group_candidates(
    map_api: MapAPI,
    point_2d: Point2D,
) -> list[int]:
    """
    Return candidate lane-group IDs at ``point_2d``.

    Containment first (cheapest correct query when ego is on a lane group), then a
    radius fallback for ego poses slightly off any polygon (e.g. parked at a curb).
    """
    contained = map_api.query_object_ids(
        geometry=point_2d.shapely_point,
        layers=[MapLayer.LANE_GROUP],
        predicate="intersects",
    )
    candidate_ids = contained.get(MapLayer.LANE_GROUP, [])
    if candidate_ids:
        return [int(id_) for id_ in candidate_ids]  # type: ignore[union-attr]

    for radius in _CANDIDATE_FALLBACK_RADII:
        nearby = map_api.get_map_objects_in_radius(
            point=point_2d,
            radius=radius,
            layers=[MapLayer.LANE_GROUP],
        )
        nearby_objects = nearby.get(MapLayer.LANE_GROUP, [])
        if nearby_objects:
            return [int(obj.object_id) for obj in nearby_objects]

    return []


def get_current_lane_group_candidates(
    ego_pose_se2: PoseSE2,
    map_api: MapAPI,
    route_lane_group_dict: dict[int, LaneGroup],
    heading_error_thresh: float = np.pi / 4,
    displacement_error_thresh: float = 3,
) -> tuple[LaneGroup, list[LaneGroup]]:
    """
    Finds the ego's current lane group and the set of nearby candidate lane groups.

    Prefers on-route candidates, falling back to the closest off-route candidate, and finally
    to any close lane group.

    Args:
        ego_pose_se2: pose of the ego vehicle
        map_api: map interface
        route_lane_group_dict: on-route lane group dictionary
        heading_error_thresh: [rad] max heading error for a lane to count as a candidate
        displacement_error_thresh: [m] max displacement error for a lane to count as a candidate

    Returns:
        tuple of (best current lane group, list of candidate lane groups)
    """
    lane_group_dict = map_api.get_map_objects_in_radius(
        point=ego_pose_se2.point_2d,
        radius=1.0,
        layers=[MapLayer.LANE_GROUP],
    )
    lane_group_candidates: list[LaneGroup] = lane_group_dict[MapLayer.LANE_GROUP]  # type: ignore[assignment]

    if len(lane_group_candidates) == 0:
        # TODO: Use query nearest once implemented in py123d.
        lane_group_dict = map_api.get_map_objects_in_radius(
            point=ego_pose_se2.point_2d,
            radius=100.0,
            layers=[MapLayer.LANE_GROUP],
        )
        lane_group_candidates = lane_group_dict[MapLayer.LANE_GROUP]  # type: ignore[assignment]

    on_route_candidates: list[LaneGroup] = []
    on_route_candidate_displacement_errors: list[float] = []
    candidates: list[LaneGroup] = []
    candidate_displacement_errors: list[float] = []

    lane_group_displacement_errors: list[float] = []
    lane_group_heading_errors: list[float] = []

    for lane_group in lane_group_candidates:
        assert isinstance(lane_group, LaneGroup), f"Expected LaneGroup, got {type(lane_group)}"
        lane_displacement_error, lane_heading_error = np.inf, np.inf

        for lane in lane_group.lanes:
            lane_discrete_poses_se2 = lane.centerline.polyline_se2.array

            lane_state_distances = np.linalg.norm(
                lane_discrete_poses_se2[..., PoseSE2Index.XY] - ego_pose_se2.point_2d.array[None, ...],
                axis=-1,
            )
            argmin = np.argmin(lane_state_distances)

            heading_error = np.abs(
                normalize_angle(
                    lane_discrete_poses_se2[argmin, PoseSE2Index.YAW] - ego_pose_se2.yaw,
                ),
            )
            displacement_error = lane_state_distances[argmin]

            if displacement_error < lane_displacement_error:
                lane_heading_error, lane_displacement_error = (
                    heading_error,
                    displacement_error,
                )

            if heading_error < heading_error_thresh and displacement_error < displacement_error_thresh:
                if lane_group.object_id in route_lane_group_dict:
                    on_route_candidates.append(lane_group)
                    on_route_candidate_displacement_errors.append(
                        displacement_error,
                    )
                else:
                    candidates.append(lane_group)
                    candidate_displacement_errors.append(displacement_error)

        lane_group_displacement_errors.append(lane_displacement_error)
        lane_group_heading_errors.append(float(lane_heading_error))

    if on_route_candidates:  # prefer on-route lane_groups
        return (
            on_route_candidates[np.argmin(on_route_candidate_displacement_errors)],
            on_route_candidates,
        )
    if candidates:  # fallback to most promising candidate
        return candidates[np.argmin(candidate_displacement_errors)], candidates

    # otherwise, just find any close lane_group
    return (
        lane_group_candidates[np.argmin(lane_group_displacement_errors)],
        lane_group_candidates,
    )


def _lane_group_entry_exit_yaw(lane_group: LaneGroup) -> tuple[float, float]:
    """
    Returns a representative ``(entry_yaw, exit_yaw)`` [rad] for a lane group.

    Uses the lane group's first lane centerline, whose start/end heading approximate the heading of
    the (near-parallel) lanes in the group at the group's entry and exit.

    Args:
        lane_group: lane group with at least one lane

    Returns:
        tuple of (entry yaw, exit yaw) in radians
    """
    poses_se2 = lane_group.lanes[0].centerline.polyline_se2.array
    return float(poses_se2[0, PoseSE2Index.YAW]), float(
        poses_se2[-1, PoseSE2Index.YAW],
    )


def _roll_out_forward_lane_groups(
    start_lane_group: LaneGroup,
    search_depth_forward: int = 30,
) -> tuple[list[LaneGroup], list[int]]:
    """
    Greedily rolls out a 'follow-the-road' route by following the straightest successor.

    At each step the successor minimizing the heading change between the current lane group's exit
    tangent and the candidate's entry tangent is chosen. Stops at ``search_depth_forward`` steps,
    when there is no successor, or when a lane group repeats (loop guard).

    Args:
        start_lane_group: lane group to start the rollout from
        search_depth_forward: max number of successor lane groups to append, defaults to 30

    Returns:
        tuple of (route lane groups, route lane group ids), starting with ``start_lane_group``
    """
    route_lane_groups: list[LaneGroup] = [start_lane_group]
    route_lane_group_ids: list[int] = [int(start_lane_group.object_id)]
    visited = {int(start_lane_group.object_id)}

    current = start_lane_group
    for _ in range(search_depth_forward):
        successors = [successor for successor in current.successors if successor.lanes]
        if not successors:
            break

        _, current_exit_yaw = _lane_group_entry_exit_yaw(current)
        next_group = min(
            successors,
            key=lambda successor: abs(
                normalize_angle(
                    _lane_group_entry_exit_yaw(successor)[0] - current_exit_yaw,
                ),
            ),
        )
        next_id = int(next_group.object_id)
        if next_id in visited:  # loop guard
            break

        route_lane_groups.append(next_group)
        route_lane_group_ids.append(next_id)
        visited.add(next_id)
        current = next_group

    return route_lane_groups, route_lane_group_ids


def infer_follow_the_road_route(
    ego_pose_se2: PoseSE2,
    map_api: MapAPI,
    search_depth_forward: int = 30,
) -> list[int]:
    """
    Synthesizes a 'follow-the-road' route from the ego's current lane group.

    Fallback used when no on-route lane groups are available (e.g. there is no logged route and the
    oracle-based inference found no path). Finds the ego's current lane group purely from the map
    (no oracle / ground-truth) and greedily follows the straightest successor along the lane-group
    graph via :func:`_roll_out_forward_lane_groups`.

    Args:
        ego_pose_se2: pose of the ego vehicle
        map_api: map interface
        search_depth_forward: max number of successor lane groups to roll out, defaults to 30

    Returns:
        ordered lane group ids starting at ego (empty if ego is not on/near any lane group)
    """
    # Guard get_current_lane_group_candidates, which calls np.argmin over the candidate list and
    # would raise on an empty list when no lane group exists near ego (e.g. ego off-map).
    nearby = map_api.get_map_objects_in_radius(
        point=ego_pose_se2.point_2d,
        radius=100.0,
        layers=[MapLayer.LANE_GROUP],
    )
    if not nearby[MapLayer.LANE_GROUP]:
        return []

    starting_group, _ = get_current_lane_group_candidates(
        ego_pose_se2=ego_pose_se2,
        map_api=map_api,
        route_lane_group_dict={},
    )
    _, route_lane_group_ids = _roll_out_forward_lane_groups(
        starting_group,
        search_depth_forward,
    )
    return route_lane_group_ids


def route_lane_group_correction(
    ego_pose_se2: PoseSE2,
    map_api: MapAPI,
    route_lane_group_dict: dict[int, LaneGroup],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
) -> tuple[list[LaneGroup], list[int]]:
    """
    Corrects and repairs the on-route lane groups for the current ego pose.

    Handles three cases: an off-route start (backward then forward graph search), disconnected
    consecutive lane groups (search for connecting links), and route loops.

    Args:
        ego_pose_se2: pose of the ego vehicle
        map_api: map interface
        route_lane_group_dict: on-route lane group dictionary
        search_depth_backward: max BFS depth for the backward search, defaults to 15
        search_depth_forward: max BFS depth for the forward search, defaults to 30

    Returns:
        tuple of (corrected lane groups, corrected lane group ids)
    """
    # TODO: Refactor code for readability

    starting_group, starting_group_candidates = get_current_lane_group_candidates(
        ego_pose_se2=ego_pose_se2,
        map_api=map_api,
        route_lane_group_dict=route_lane_group_dict,
    )
    starting_block_ids = [lane_group.object_id for lane_group in starting_group_candidates]

    route_lane_groups = list(route_lane_group_dict.values())
    route_lane_group_ids = list(route_lane_group_dict.keys())

    # When no route is provided, synthesize a follow-the-road route from the current lane group.
    # Avoids indexing the empty route below and keeps the agent driving down the road.
    if not route_lane_group_ids:
        return _roll_out_forward_lane_groups(
            starting_group,
            search_depth_forward,
        )

    # Fix 1: when agent starts off-route
    if starting_group.object_id not in route_lane_group_ids:
        # Backward search if current lane_group not in route
        graph_search = BreadthFirstSearchLaneGroup(
            route_lane_group_ids[0],
            map_api,
            forward_search=False,
        )
        path, path_id, path_found = graph_search.search(
            starting_block_ids,  # type: ignore[arg-type]
            max_depth=search_depth_backward,
        )

        if path_found:
            route_lane_groups[:0] = path[:-1]
            route_lane_group_ids[:0] = path_id[:-1]

        else:
            # Forward search to any route lane_group
            graph_search = BreadthFirstSearchLaneGroup(
                int(starting_group.object_id),
                map_api,
                forward_search=True,
            )
            path, path_id, path_found = graph_search.search(
                route_lane_group_ids[:3],
                max_depth=search_depth_forward,
            )

            if path_found:
                end_lane_group_idx = np.argmax(
                    np.array(route_lane_group_ids) == path_id[-1],
                )

                route_lane_groups = route_lane_groups[end_lane_group_idx + 1 :]
                route_lane_group_ids = route_lane_group_ids[end_lane_group_idx + 1 :]

                route_lane_groups[:0] = path
                route_lane_group_ids[:0] = path_id

    # Fix 2: check if lane_groups are linked, search for links if not
    lane_groups_to_append: dict[int, tuple[list[LaneGroup], list[int]]] = {}
    for i in range(len(route_lane_groups) - 1):
        next_incoming_block_ids = [_lane_group.object_id for _lane_group in route_lane_groups[i + 1].predecessors]
        is_incoming = route_lane_group_ids[i] in next_incoming_block_ids

        if is_incoming:
            continue

        graph_search = BreadthFirstSearchLaneGroup(
            route_lane_group_ids[i],
            map_api,
            forward_search=True,
        )
        path, path_id, path_found = graph_search.search(
            route_lane_group_ids[i + 1],
            max_depth=search_depth_forward,
        )  # type: ignore[arg-type]

        if path_found and path and len(path) >= 3:
            path, path_id = path[1:-1], path_id[1:-1]
            lane_groups_to_append[i] = (path, path_id)

    # append missing intermediate lane_groups
    offset = 1
    for i, (path, path_id) in lane_groups_to_append.items():
        route_lane_groups[i + offset : i + offset] = path
        route_lane_group_ids[i + offset : i + offset] = path_id
        offset += len(path)

    # Fix 3: cut route-loops
    route_lane_groups, route_lane_group_ids = remove_route_loops(
        route_lane_groups,
        route_lane_group_ids,
    )

    return route_lane_groups, route_lane_group_ids


def remove_route_loops(
    route_lane_groups: list[LaneGroup],
    route_lane_group_ids: list[int],
) -> tuple[list[LaneGroup], list[int]]:
    """
    Removes the end of the route where a lane group intersects an earlier one (forming a loop).

    Args:
        route_lane_groups: input route lane_groups
        route_lane_group_ids: input route lane_groups ids

    Returns:
        tuple of ids and lane_groups of route without loops
    """

    lane_group_intersection_dict: dict[int, geom.base.BaseGeometry] = {}

    loop_idx: int | None = None

    for idx, lane_group in enumerate(route_lane_groups):
        # loops only occur at intersection, thus searching for lane_group-connectors.
        intersection: Intersection | None = lane_group.intersection
        if intersection is not None:
            if len(lane_group_intersection_dict) == 0:
                for intersection_lane_group in intersection.lane_groups:
                    lane_group_intersection_dict[int(intersection_lane_group.object_id)] = (
                        intersection_lane_group.shapely_polygon
                    )
                continue
            occupancy_map = OccupancyMap2D.from_dict(
                lane_group_intersection_dict,
            )
            intersecting_ids = occupancy_map.intersects(
                lane_group.shapely_polygon,
            )
            for intersecting_id in intersecting_ids:
                intersecting_polygon = lane_group_intersection_dict[int(intersecting_id)]
                area = intersecting_polygon.intersection(
                    lane_group.shapely_polygon,
                ).area
                if area > 1.0:
                    loop_idx = idx
                    break

    if loop_idx:
        route_lane_groups = route_lane_groups[:loop_idx]
        route_lane_group_ids = route_lane_group_ids[:loop_idx]

    return route_lane_groups, route_lane_group_ids
