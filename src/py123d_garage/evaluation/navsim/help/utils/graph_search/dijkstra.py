# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

import numpy as np
from py123d.datatypes import Lane, LaneGroup


class Dijkstra:
    """
    Performs Dijkstra's shortest-path search on a lane-level graph.

    The goal condition is met when a lane is found in the target lane_group.
    """

    def __init__(self, start_lane: Lane, candidate_lane_ids: list[int]):
        """
        Constructor for the Dijkstra class.

        Args:
            start_lane: The starting lane for the search
            candidate_lane_ids: The candidate lane ids that can be included in the search.
        """
        self._queue: list[Lane] = [start_lane]
        self._parent: dict[int, Lane | None] = {}
        self._candidate_lane_ids = candidate_lane_ids

    def search(self, target_lane_group: LaneGroup) -> tuple[list[Lane], bool]:
        """
        Performs dijkstra's shortest path to find a route to the target lane_group.

        Args:
            target_lane_group: The target lane_group the path should end at.

        Returns:
            - A route starting from the given start lane - A bool indicating if the route is successfully found. Successful means that there exists a path from the start lane to a lane contained in the target lane_group. If unsuccessful the shortest deepest path is returned.
        """
        start_lane = self._queue[0]

        # Initial search states
        path_found: bool = False
        end_lane: Lane = start_lane

        self._parent[int(start_lane.object_id)] = None
        self._frontier: list[int] = [int(start_lane.object_id)]
        self._dist: list[float] = [1]
        self._depth: list[int] = [1]

        self._expanded: list[Lane] = []
        self._expanded_id: list[int] = []
        self._expanded_dist: list[float] = []
        self._expanded_depth: list[int] = []

        while len(self._queue) > 0:
            dist, idx = min((val, idx) for (idx, val) in enumerate(self._dist))
            current_lane = self._queue[idx]
            current_depth = self._depth[idx]

            del (
                self._dist[idx],
                self._queue[idx],
                self._frontier[idx],
                self._depth[idx],
            )

            if self._check_goal_condition(current_lane, target_lane_group):
                end_lane = current_lane
                path_found = True
                break

            self._expanded.append(current_lane)
            self._expanded_id.append(int(current_lane.object_id))
            self._expanded_dist.append(dist)
            self._expanded_depth.append(current_depth)

            # Populate queue
            for next_lane in current_lane.successors:
                next_lane_id = int(next_lane.object_id)
                if next_lane_id not in self._candidate_lane_ids:
                    continue

                alt = dist + self._edge_cost(next_lane)
                if next_lane_id not in self._expanded_id and next_lane_id not in self._frontier:
                    self._parent[next_lane_id] = current_lane
                    self._queue.append(next_lane)
                    self._frontier.append(next_lane_id)
                    self._dist.append(alt)
                    self._depth.append(current_depth + 1)
                    end_lane = next_lane

                elif next_lane_id in self._frontier:
                    next_lane_idx = self._frontier.index(next_lane_id)
                    current_cost = self._dist[next_lane_idx]
                    if alt < current_cost:
                        self._parent[next_lane_id] = current_lane
                        self._dist[next_lane_idx] = alt
                        self._depth[next_lane_idx] = current_depth + 1

        if not path_found:
            # filter max depth
            max_depth = max(self._expanded_depth)
            idx_max_depth = [
                int(idx)
                for idx in np.where(
                    np.array(self._expanded_depth) == max_depth,
                )[0]
            ]
            dist_at_max_depth = [self._expanded_dist[i] for i in idx_max_depth]

            dist, _idx = min((val, idx) for (idx, val) in enumerate(dist_at_max_depth))
            end_lane = self._expanded[idx_max_depth[_idx]]

        return self._construct_path(end_lane), path_found

    @staticmethod
    def _edge_cost(lane: Lane) -> float:
        """
        Edge cost of given lane.

        Args:
            lane: lane class

        Returns:
            length of lane centerline
        """
        return lane.centerline.length

    @staticmethod
    def _check_goal_condition(
        current_lane: Lane,
        target_lane_group: LaneGroup,
    ) -> bool:
        """
        Check if the current lane is at the target lane_group.

        Args:
            current_lane: The lane to check.
            target_lane_group: The target lane_group the lane should be contained in.

        Returns:
            whether the current lane is in the target lane_group
        """
        assert current_lane.lane_group_id is not None, f"Lane {current_lane.object_id} has no lane_group_id."
        return int(current_lane.lane_group_id) == int(
            target_lane_group.object_id,
        )

    def _construct_path(self, end_lane: Lane) -> list[Lane]:
        """
        Reconstructs the path by back-propagating parents from the end lane to the start lane.

        Args:
            end_lane: The end lane to start back propagating back to the start lane.

        Returns:
            The constructed path as a list of Lane
        """
        path = [end_lane]
        while True:
            node = self._parent[int(end_lane.object_id)]
            if node is None:
                break
            path.append(node)
            end_lane = node
        path.reverse()

        return path
