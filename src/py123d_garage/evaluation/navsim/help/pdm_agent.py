# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from py123d.api import MapAPI, SceneAPI
from py123d.datatypes import EgoStateSE2, Lane, LaneGroup
from py123d.geometry import OccupancyMap2D, PolylineSE2

from py123d_garage.datatypes.trajectory import (
    TrajectorySE2,
)
from py123d_garage.evaluation.navsim.help.observation.pdm_observation import (
    PDMObservation,
)
from py123d_garage.evaluation.navsim.help.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from py123d_garage.evaluation.navsim.help.proposal.pdm_generator import (
    PDMGenerator,
)
from py123d_garage.evaluation.navsim.help.proposal.pdm_proposal import (
    PDMProposalManager,
)
from py123d_garage.evaluation.navsim.help.scoring.pdm_scorer import PDMScorer
from py123d_garage.evaluation.navsim.help.simulation.pdm_simulator import (
    PDMSimulator,
)
from py123d_garage.evaluation.navsim.help.trajectory_grid import TrajectoryGrid
from py123d_garage.evaluation.navsim.help.utils.pdm_closed_utils import (
    build_drivable_area_occupancy_map,
    build_route_dicts,
    correct_route_lane_groups,
    get_centerline_as_polyline_se2,
    get_proposal_paths,
    get_starting_lane,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)
from py123d_garage.evaluation.navsim.help.utils.route_utils import (
    get_route_lane_group_ids,
    infer_follow_the_road_route,
)

LOG = logging.getLogger(__name__)


class PDMAgent:
    """PDM-Closed planner."""

    def __init__(
        self,
        trajectory_grid: TrajectoryGrid | None = None,
        proposal_grid: TrajectoryGrid | None = None,
        idm_policies: BatchIDMPolicy | None = None,
        lateral_offsets: Sequence[float] | None = (-1, 1),
        map_radius: float = 200,
        route_correction: bool = True,
    ):
        """
        Constructor for PDMAgent.

        Args:
            trajectory_grid: Sampling parameters for final trajectory
            proposal_grid: Sampling parameters for proposals
            idm_policies: BatchIDMPolicy class
            lateral_offsets: centerline offsets for proposals (optional)
            map_radius: radius around ego to consider
            route_correction: whether to correct the route lane groups on the first iteration
        """
        if trajectory_grid is None:
            trajectory_grid = TrajectoryGrid.from_num_poses_and_interval(
                num_poses=80,
                interval_us=100_000,
            )
        if proposal_grid is None:
            proposal_grid = TrajectoryGrid.from_num_poses_and_interval(
                num_poses=40,
                interval_us=100_000,
            )
        if idm_policies is None:
            idm_policies = BatchIDMPolicy()
        assert trajectory_grid.interval_us == proposal_grid.interval_us, (
            "PDMClosedPlanner: Proposals and Trajectory must have equal interval length!"
        )

        # config parameters
        self._trajectory_grid: TrajectoryGrid = trajectory_grid
        self._proposal_grid: TrajectoryGrid = proposal_grid
        self._idm_policies: BatchIDMPolicy = idm_policies
        self._lateral_offsets: Sequence[float] | None = lateral_offsets
        self._map_radius: float = map_radius
        self._route_correction: bool = route_correction

        # observation/forecasting class
        self._observation = PDMObservation(
            trajectory_grid,
            proposal_grid,
            map_radius,
        )

        # proposal/trajectory related classes
        self._generator = PDMGenerator(
            trajectory_grid,
            proposal_grid,
        )
        self._simulator = PDMSimulator(proposal_grid)
        self._scorer = PDMScorer(proposal_grid)
        self._emergency_brake = PDMEmergencyBrake(trajectory_grid)

        self._iteration: int = 0

        # lazy loaded
        self._map_api: MapAPI | None = None
        self._route_lane_group_dict: dict[int, LaneGroup] | None = None
        self._route_lane_dict: dict[int, Lane] | None = None
        self._centerline: PolylineSE2 | None = None
        self._drivable_area_map: OccupancyMap2D | None = None
        self._proposal_manager: PDMProposalManager | None = None

    def initialize(self) -> None:
        """Resets the planner state before a run."""
        self._iteration = 0

    def compute_trajectory(self, scene_api: SceneAPI) -> TrajectorySE2:
        """Runs PDM-Closed planning for one scene; returns the trajectory in the absolute/global frame."""

        ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
        box_detections_se3 = scene_api.get_box_detections_se3_at_iteration(0)
        traffic_light_detections = scene_api.get_traffic_light_detections_at_iteration(0)
        if self._iteration == 0:
            self._map_api = scene_api.get_map_api()
            self._route_lane_group_ids = get_route_lane_group_ids(scene_api)
            assert self._map_api is not None, "Map API not found in Scene API."
            assert self._route_lane_group_ids is not None, "Route lane group ids not found in Scene API."

            if len(self._route_lane_group_ids) == 0:
                # No on-route lane groups available (no logged route and oracle inference found no
                # path). Synthesize a follow-the-road route from ego's current lane group.
                assert ego_state_se3 is not None, "Ego state modality not found at iteration."
                self._route_lane_group_ids = infer_follow_the_road_route(
                    ego_state_se3.ego_state_se2.rear_axle_se2,
                    self._map_api,
                )
                LOG.warning(
                    f"No on-route lane groups available; using follow-the-road "
                    f"fallback route ({len(self._route_lane_group_ids)} lane groups).",
                )

        assert ego_state_se3 is not None, "Ego state modality not found at iteration."
        assert box_detections_se3 is not None, "Box detections modality not found at iteration."
        assert self._map_api is not None, "Map API not found in Scene API."
        assert self._route_lane_group_ids is not None, "Route lane group ids not found in Scene API."

        # Safe-stop last resort: ego is not on/near any lane group, so no route can be built.
        # FIXME: Remove this fallback again or improve the breaking trajectory.
        if len(self._route_lane_group_ids) == 0:
            LOG.warning(
                "No lane group near ego; returning a decelerate-to-stop trajectory.",
            )
            self._iteration += 1
            return self._emergency_brake.generate_stop_trajectory(
                ego_state_se3.ego_state_se2,
            )

        self._route_lane_group_dict, self._route_lane_dict = build_route_dicts(
            self._map_api,
            self._route_lane_group_ids,
        )
        gc.disable()

        assert self._map_api is not None, (
            "Planner not initialized properly. Call initialize() before compute_planner_trajectory()."
        )
        assert self._route_lane_group_dict is not None, (
            "Planner not initialized properly. Call initialize() before compute_planner_trajectory()."
        )

        ego_state_se2 = ego_state_se3.ego_state_se2
        box_detections_se2 = box_detections_se3.box_detections_se2

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            assert self._map_api is not None, "Planner not initialized properly."
            assert self._route_lane_group_dict is not None, "Planner not initialized properly."
            if self._route_correction:
                self._route_lane_group_dict, self._route_lane_dict = correct_route_lane_groups(
                    ego_state_se2=ego_state_se2,
                    map_api=self._map_api,
                    route_lane_group_dict=self._route_lane_group_dict,
                )

        assert self._route_lane_group_dict is not None, "Route lane dicts not initialized."
        assert self._route_lane_dict is not None, "Route lane dicts not initialized."

        # Update/Create drivable area polygon map
        self._drivable_area_map = build_drivable_area_occupancy_map(
            map_api=self._map_api,
            ego_state_se2=ego_state_se2,
            map_radius=self._map_radius,
        )

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state_se2=ego_state_se2,
            box_detections_se2=box_detections_se2,
            traffic_light_detections=traffic_light_detections,
            route_lane_dict=self._route_lane_dict,
        )

        # TODO: Refactor the rest and re-integrate the following steps:
        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(ego_state_se2)

        # 3. Generate/Unroll proposals
        assert self._proposal_manager is not None, "Proposal manager not initialized."
        proposals_array = self._generator.generate_proposals(
            ego_state_se2,
            self._observation,
            self._proposal_manager,
        )

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array,
            ego_state_se2,
        )

        # 5. Score proposals
        assert self._centerline is not None, "Centerline not initialized."
        pdm_results = self._scorer.score_proposals(
            states=simulated_proposals_array,
            observation=self._observation,
            centerline=self._centerline,
            route_lane_ids=list(self._route_lane_group_dict.keys()),
            drivable_area_map=self._drivable_area_map,
            ego_metadata=ego_state_se2.metadata,
        )
        proposal_scores: npt.NDArray[np.float64] = np.asarray(
            pd.concat(pdm_results)["pdm_score"],
            dtype=np.float64,
        )

        trajectory = self._generator.generate_trajectory(
            int(np.argmax(proposal_scores)),
        )

        self._iteration += 1
        return trajectory

    def _update_proposal_manager(self, ego_state_se2: EgoStateSE2) -> None:
        """
        Updates or initializes the PDMProposalManager.

        Args:
            ego_state_se2: state of ego-vehicle
        """
        assert self._route_lane_dict is not None, "Planner not initialized properly."
        assert self._drivable_area_map is not None, "Planner not initialized properly."
        current_lane = get_starting_lane(
            ego_state_se2,
            self._route_lane_dict,
            self._drivable_area_map,
        )

        # TODO: Find additional conditions to trigger re-planning
        create_new_proposals = self._iteration == 0
        if create_new_proposals:
            assert self._route_lane_group_dict is not None, "Planner not initialized properly."
            self._centerline = get_centerline_as_polyline_se2(
                current_lane,
                self._route_lane_group_dict,
                self._route_lane_dict,
                ego_state_se2=ego_state_se2,
            )
            proposal_paths: list[PolylineSE2] = get_proposal_paths(
                self._centerline,
                self._lateral_offsets,
            )
            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        assert self._proposal_manager is not None, "Proposal manager not initialized."
        self._proposal_manager.update(current_lane.speed_limit_mps)
