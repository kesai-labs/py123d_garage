# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from collections import deque
from typing import cast

from py123d.api import MapAPI
from py123d.datatypes import LaneGroup, MapLayer


class BreadthFirstSearchLaneGroup:
    """Performs iterative breadth-first search on the lane_group graph."""

    def __init__(
        self,
        start_lane_group_id: int,
        map_api: MapAPI,
        forward_search: bool = True,
    ):
        """
        Sets up the search from a start lane group.

        Args:
            start_lane_group_id: ID of the lane group to start from.
            map_api: Map holding the lane group graph.
            forward_search: Whether to search along successors instead of predecessors.
        """
        self._map_api: MapAPI = map_api

        _start_lane_group = self._map_api.get_map_object_in_layer(
            start_lane_group_id,
            MapLayer.LANE_GROUP,
        )
        assert _start_lane_group is not None, f"LaneGroup with id {start_lane_group_id} not found in map."
        self._queue: deque[LaneGroup | None] = deque(
            [cast(LaneGroup, _start_lane_group), None],
        )
        self._parent: dict[str, LaneGroup | None] = {}
        self._forward_search: bool = forward_search

    def search(
        self,
        target_lane_group_id: int | list[int],
        max_depth: int,
    ) -> tuple[list[LaneGroup], list[int], bool]:
        """
        Searches the shortest lane group path to any of the target lane groups.

        Args:
            target_lane_group_id: ID or IDs of the target lane groups.
            max_depth: Maximum search depth.

        Returns:
            The path lane groups, their IDs and whether a target was reached.
        """
        if isinstance(target_lane_group_id, int):
            target_lane_group_ids = [target_lane_group_id]
        else:
            target_lane_group_ids = list(set(target_lane_group_id))

        start_edge: LaneGroup = self._queue[0]  # type: ignore[assignment]

        path_found: bool = False
        end_edge: LaneGroup = start_edge  # type: ignore[assignment]
        end_depth: int = 1
        depth: int = 1

        self._parent[str(start_edge.object_id) + f"_{depth}"] = None

        while self._queue:
            current_edge = self._queue.popleft()

            if self._check_end_condition(depth, max_depth):
                break

            # None marks the end of a depth level.
            if current_edge is None:
                depth += 1
                self._queue.append(None)
                if self._queue[0] is None:
                    break
                continue

            if self._check_goal_condition(
                current_edge,
                target_lane_group_ids,
                depth,
                max_depth,
            ):
                end_edge = current_edge
                end_depth = depth
                path_found = True
                break

            neighbors = current_edge.successors if self._forward_search else current_edge.predecessors

            for next_edge in neighbors:
                self._queue.append(next_edge)
                self._parent[str(next_edge.object_id) + f"_{depth + 1}"] = current_edge
                end_edge = next_edge
                end_depth = depth + 1

        path, path_id = self._construct_path(end_edge, end_depth)
        return path, path_id, path_found

    @staticmethod
    def _check_end_condition(depth: int, max_depth: int) -> bool:
        """
        Checks whether the search exceeded its maximum depth.

        Args:
            depth: The current depth to check.
            max_depth: The maximum depth to check against.

        Returns:
            whether depth exceeds the target depth.
        """
        return depth > max_depth

    def _check_goal_condition(
        self,
        current_edge: LaneGroup,
        target_lane_group_ids: list[int],
        depth: int,
        max_depth: int,
    ) -> bool:
        """
        Checks whether the current lane group is a target within the maximum depth.

        Args:
            current_edge: edge to check.
            target_lane_group_ids: list of target lane_group ids.
            depth: current depth to check.
            max_depth: maximum depth the edge should be at.

        Returns:
            True if the lane edge is contained in the target lane_group, False otherwise.
        """
        return int(current_edge.object_id) in target_lane_group_ids and depth <= max_depth

    def _construct_path(
        self,
        end_edge: LaneGroup,
        depth: int,
    ) -> tuple[list[LaneGroup], list[int]]:
        """
        Reconstructs the path to the end lane group by following the parents back to the start.

        Args:
            end_edge: The end edge to start back propagating back to the start edge.
            depth: The depth of the target edge.

        Returns:
            The path lane groups and their IDs in driving order.
        """
        _end_edge = end_edge
        path = [_end_edge]
        path_id = [_end_edge.object_id]

        while self._parent[str(_end_edge.object_id) + f"_{depth}"] is not None:  # type: ignore[arg-type]
            path.append(self._parent[str(_end_edge.object_id) + f"_{depth}"])  # type: ignore[arg-type]
            path_id.append(path[-1].object_id)
            _end_edge = self._parent[str(_end_edge.object_id) + f"_{depth}"]  # type: ignore[arg-type]
            depth -= 1

        if self._forward_search:
            path.reverse()
            path_id.reverse()

        return (path, path_id)  # type: ignore[return-value]
