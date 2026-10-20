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

# Search radii in meters, tried in order, for lane groups near a point outside every lane group.
_CANDIDATE_FALLBACK_RADII: list[float] = [5.0, 20.0, 50.0]
# Whether nuPlan scenes use their logged route instead of an inferred one.
_USE_NUPLAN_DEFAULT_ROUTE: bool = True
# How far ahead in the log, in seconds, the inferred route leads to.
_TARGET_ROUTE_LOOKAHEAD_TIME_S: float = 60.0


def get_route_lane_group_ids(scene_api: SceneAPI) -> list[int]:
    """
    Gets the route of a scene, the logged nuPlan route if available, else one inferred from the log.

    Args:
        scene_api: Scene to get the route of.

    Returns:
        The ordered on-route lane group IDs, empty if none can be determined.
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
    Infers the route as the shortest lane group path from ego to its logged pose _TARGET_ROUTE_LOOKAHEAD_TIME_S ahead.

    Args:
        scene_api: Scene to infer the route of.

    Returns:
        The lane group IDs along the route, empty if no path is found.
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
    Finds the lane groups containing a point, else the lane groups within the smallest fallback radius.

    Args:
        map_api: Map to query.
        point_2d: Point to query at.

    Returns:
        The candidate lane group IDs, empty if none is near.
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
    Finds ego's current lane group, preferring aligned on-route lane groups, then aligned ones, then the closest.

    Args:
        ego_pose_se2: Ego pose.
        map_api: Map of the scene.
        route_lane_group_dict: On-route lane groups, keyed by ID.
        heading_error_thresh: [rad] Maximum heading error for a lane to count as a candidate.
        displacement_error_thresh: [m] Maximum displacement error for a lane to count as a candidate.

    Returns:
        The current lane group and the candidate lane groups.
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
    Approximates a lane group's entry and exit heading by its first lane's centerline.

    Args:
        lane_group: Lane group with at least one lane.

    Returns:
        The entry and exit yaw in radians.
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
    Builds a follow-the-road route by greedily taking the successor with the smallest heading change.

    Args:
        start_lane_group: Lane group to start from.
        search_depth_forward: Maximum number of successor lane groups to append.

    Returns:
        The route lane groups and their IDs, starting with start_lane_group.
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
    Builds a follow-the-road route from ego's current lane group, using the map only.

    Args:
        ego_pose_se2: Ego pose.
        map_api: Map of the scene.
        search_depth_forward: Maximum number of successor lane groups to append.

    Returns:
        The ordered lane group IDs starting at ego, empty if no lane group is near ego.
    """
    # get_current_lane_group_candidates raises when no lane group is near ego.
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
    Repairs the route for the current ego pose: an off-route start, unlinked lane groups and loops.

    Args:
        ego_pose_se2: Ego pose.
        map_api: Map of the scene.
        route_lane_group_dict: On-route lane groups, keyed by ID.
        search_depth_backward: Maximum BFS depth of the backward search.
        search_depth_forward: Maximum BFS depth of the forward search.

    Returns:
        The corrected lane groups and their IDs.
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

    # The code below indexes the route, so an empty route falls back to following the road.
    if not route_lane_group_ids:
        return _roll_out_forward_lane_groups(
            starting_group,
            search_depth_forward,
        )

    # Fix 1: when agent starts off-route
    if starting_group.object_id not in route_lane_group_ids:
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
        route_lane_groups: Route lane groups.
        route_lane_group_ids: IDs of the route lane groups.

    Returns:
        The route lane groups and their IDs, cut before the loop.
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
