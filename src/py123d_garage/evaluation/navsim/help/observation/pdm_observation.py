# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from typing import cast

import numpy as np
from py123d.api import SceneAPI
from py123d.datatypes import (
    BoxDetectionSE2,
    BoxDetectionsSE2,
    EgoStateSE2,
    Lane,
    TrafficLightDetections,
    TrafficLightStatus,
)
from py123d.geometry import OccupancyMap2D
from py123d.geometry.utils.bounding_box_utils import (
    bbse2_array_to_corners_array,
    corners_2d_array_to_polygon_array,
)
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from py123d_garage.evaluation.navsim.help.observation.pdm_object_manager import (
    PDMObjectManager,
)
from py123d_garage.evaluation.navsim.help.trajectory_grid import TrajectoryGrid


class PDMObservation:
    """PDM's observation class for forecasted occupancy maps."""

    def __init__(
        self,
        trajectory_grid: TrajectoryGrid,
        proposal_grid: TrajectoryGrid,
        map_radius: float | None = 50,
        observation_sample_res: int = 2,
        extend_observation_for_ttc: bool = True,
    ):
        """
        Constructor of PDMObservation.

        Args:
            trajectory_grid: Sampling parameters for final trajectory
            proposal_grid: Sampling parameters for proposals
            map_radius: radius around ego to consider, defaults to 50
            observation_sample_res: sample resolution of forecast, defaults to 2
            extend_observation_for_ttc: extend observation for TTC metric, defaults to True
        """
        assert trajectory_grid.interval_us == proposal_grid.interval_us, (
            "PDMObservation: Proposals and Trajectory must have equal interval length!"
        )

        # observation needs length of trajectory horizon or proposal horizon +1s (for TTC metric)
        self._sample_interval: float = trajectory_grid.interval_s

        if extend_observation_for_ttc:
            self._observation_samples: int = max(
                proposal_grid.num_poses + int(1 / self._sample_interval),
                trajectory_grid.num_poses,
            )
        else:
            self._observation_samples = max(
                trajectory_grid.num_poses,
                proposal_grid.num_poses,
            )

        self._map_radius: float | None = map_radius
        self._observation_sample_res: int = observation_sample_res

        # useful things
        self._global_to_local_idcs = [
            idx // observation_sample_res for idx in range(self._observation_samples + observation_sample_res)
        ]
        self._collided_track_ids: list[str] = []
        self._red_light_token = "red_light"

        # lazy loaded (during update)
        self._occupancy_maps: list[OccupancyMap2D] = []
        self._unique_objects: dict[str, BoxDetectionSE2] | None = None

    def __getitem__(self, time_idx: int) -> OccupancyMap2D:
        """
        Retrieves the occupancy map for time_idx, adapting the temporal resolution.

        Args:
            time_idx: index for future simulation iterations [10Hz]

        Returns:
            occupancy map
        """
        assert len(self._occupancy_maps) > 0, "PDMObservation: Has not been updated yet!"
        assert 0 <= time_idx < len(self._global_to_local_idcs), f"PDMObservation: index {time_idx} out of range!"

        local_idx = self._global_to_local_idcs[time_idx]
        return self._occupancy_maps[local_idx]

    @property
    def collided_track_ids(self) -> list[str]:
        """Track tokens of objects ego has already collided with."""
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._collided_track_ids

    @property
    def red_light_token(self) -> str:
        """Token prefix used to identify red-light occupancy entries."""
        return self._red_light_token

    @property
    def unique_objects(self) -> dict[str, BoxDetectionSE2]:
        """Mapping from track token to the tracked object."""
        assert self._unique_objects is not None, "PDMObservation: Has not been updated yet!"
        return self._unique_objects

    @property
    def box_detections_se2(self) -> BoxDetectionsSE2:
        """Box detections from the most recent update."""
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._box_detections_se2

    def update(
        self,
        ego_state_se2: EgoStateSE2,
        box_detections_se2: BoxDetectionsSE2,
        traffic_light_detections: TrafficLightDetections | None,
        route_lane_dict: dict[int, Lane],
    ) -> None:
        """
        Forecasts object occupancy and lazily updates the PDMObservation state.

        Args:
            ego_state_se2: state of ego vehicle
            box_detections_se2: input box detections
            traffic_light_detections: list of traffic light states
            route_lane_dict: dictionary of on-route lanes
        """

        if traffic_light_detections is None:
            traffic_light_detections = TrafficLightDetections(
                [],
                ego_state_se2.timestamp,
            )

        self._occupancy_maps = []
        object_manager = self._get_object_manager(
            ego_state_se2,
            box_detections_se2,
        )

        (
            traffic_light_tokens,
            traffic_light_polygons,
        ) = self._get_traffic_light_geometries(
            traffic_light_detections,
            route_lane_dict,
        )

        (
            static_object_tokens,
            static_object_bbse2,
            dynamic_object_tokens,
            dynamic_object_bbse2,
            dynamic_object_dxy,
        ) = object_manager.get_nearest_objects(ego_state_se2.center_2d)

        has_static_object, has_dynamic_object = (
            len(static_object_tokens) > 0,
            len(dynamic_object_tokens) > 0,
        )

        if has_static_object:
            static_object_polygons = corners_2d_array_to_polygon_array(
                bbse2_array_to_corners_array(static_object_bbse2),
            )

        else:
            static_object_polygons = np.array([], dtype=np.object_)

        if has_dynamic_object:
            dynamic_object_corners = bbse2_array_to_corners_array(
                dynamic_object_bbse2,
            )
        else:
            dynamic_object_corners = np.array([], dtype=np.float64)
            dynamic_object_polygons = np.array([], dtype=np.object_)
            dynamic_object_tokens = []

        traffic_light_polygons = np.array(
            traffic_light_polygons,
            dtype=np.object_,
        )

        for sample in np.arange(
            0,
            self._observation_samples + self._observation_sample_res,
            self._observation_sample_res,
        ):
            if has_dynamic_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_object_corners_t = dynamic_object_corners + delta_t * dynamic_object_dxy[:, None]
                dynamic_object_polygons = corners_2d_array_to_polygon_array(
                    dynamic_object_corners_t,
                )
            else:
                dynamic_object_polygons = np.array([], dtype=np.object_)

            all_polygons = np.concatenate(
                [
                    static_object_polygons,
                    dynamic_object_polygons,
                    traffic_light_polygons,
                ],
                axis=0,
            )

            occupancy_map = OccupancyMap2D(
                geometries=all_polygons,  # type: ignore[arg-type]
                ids=static_object_tokens + dynamic_object_tokens + traffic_light_tokens,
            )
            self._occupancy_maps.append(occupancy_map)

        # save collided objects to ignore in the future
        ego_polygon: Polygon = ego_state_se2.bounding_box_se2.shapely_polygon
        # The occupancy maps built above are keyed by string track tokens.
        intersecting_obstacles = cast(
            list[str],
            self._occupancy_maps[0].intersects(ego_polygon),
        )
        new_collided_track_ids: list[str] = []

        for intersecting_obstacle in intersecting_obstacles:
            if str(self._red_light_token) in intersecting_obstacle:
                within = ego_polygon.within(
                    self._occupancy_maps[0][intersecting_obstacle],
                )
                if not within:
                    continue
            new_collided_track_ids.append(intersecting_obstacle)

        self._box_detections_se2 = box_detections_se2
        self._collided_track_ids += new_collided_track_ids
        self._unique_objects = object_manager.unique_objects
        self._initialized = True

    def update_replay(self, scene_api: SceneAPI) -> None:
        """
        Builds occupancy maps directly from logged future detections (privileged replay).

        Unlike :func:`update`, this reads ground-truth box detections at each future iteration
        instead of forecasting object motion from the current frame.

        Args:
            scene_api: API providing access to logged future box detections.
        """
        occupancy_maps: list[OccupancyMap2D] = []
        unique_objects: dict[str, BoxDetectionSE2] = {}

        max_available_iteration = scene_api.number_of_iterations - 1
        max_replay_iteration = min(
            self._observation_samples,
            max_available_iteration,
        )

        for iteration in range(max_replay_iteration + 1):
            _box_detections_se3 = scene_api.get_box_detections_se3_at_iteration(
                iteration,
            )
            assert _box_detections_se3 is not None, (
                f"PDMObservation: Missing box detections at iteration {iteration} for replay update!"
            )
            _box_detections_se2 = _box_detections_se3.box_detections_se2
            _occupancy_dict: dict[str, BaseGeometry] = {}
            for box_detection_se2 in _box_detections_se2:
                token = box_detection_se2.attributes.track_token
                polygon = box_detection_se2.shapely_polygon
                _occupancy_dict[token] = polygon
                if token not in unique_objects:
                    unique_objects[token] = box_detection_se2
            occupancy_maps.append(OccupancyMap2D.from_dict(_occupancy_dict))

        assert len(occupancy_maps) > 0, "PDMObservation: Replay update produced no occupancy maps."

        if len(occupancy_maps) < self._observation_samples + 1:
            # Some scenes provide a shorter logged future horizon than required by PDM scoring.
            # Reuse the last available occupancy map for the remaining horizon.
            last_map = occupancy_maps[-1]
            occupancy_maps.extend(
                [last_map] * (self._observation_samples + 1 - len(occupancy_maps)),
            )

        self._occupancy_maps = occupancy_maps
        self._collided_track_ids = []
        self._unique_objects = unique_objects
        self._initialized = True

    def _get_object_manager(
        self,
        ego_state_se2: EgoStateSE2,
        box_detections_se2: BoxDetectionsSE2,
    ) -> PDMObjectManager:
        """
        Creates an object manager populated with the valid tracked objects.

        Objects beyond the map radius or already collided with are skipped.

        Args:
            ego_state_se2: state of ego-vehicle of initial step
            box_detections_se2: input box detections of initial step

        Returns:
            PDMObjectManager class
        """
        object_manager = PDMObjectManager()

        for box_detection_se2 in box_detections_se2:
            ego_box_distance = np.linalg.norm(
                ego_state_se2.center_2d.array - box_detection_se2.center_se2.point_2d.array,
            )
            if (self._map_radius is not None and ego_box_distance > self._map_radius) or (
                box_detection_se2.attributes.track_token in self._collided_track_ids
            ):
                continue

            object_manager.add_object(box_detection_se2)

        return object_manager

    def _get_traffic_light_geometries(
        self,
        traffic_light_detections: TrafficLightDetections,
        route_lane_dict: dict[int, Lane],
    ) -> tuple[list[str], list[Polygon]]:
        """
        Collects red traffic lights along ego's route.

        Args:
            traffic_light_detections: wrapper class for traffic light detections in 123D.
            route_lane_dict: dictionary of on-route lanes

        Returns:
            tuple of tokens and polygons of red traffic lights
        """
        traffic_light_tokens: list[str] = []
        traffic_light_polygons: list[Polygon] = []

        for data in traffic_light_detections:
            lane_id = int(data.lane_id)

            if (data.status == TrafficLightStatus.RED) and (lane_id in route_lane_dict):
                lane = route_lane_dict[lane_id]
                traffic_light_tokens.append(
                    f"{self._red_light_token}_{lane_id}",
                )
                traffic_light_polygons.append(lane.shapely_polygon)

        return traffic_light_tokens, traffic_light_polygons
