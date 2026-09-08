# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import (
    BoxDetectionSE2,
    EgoStateSE2,
    EgoStateSE3Metadata,
    Timestamp,
)
from py123d.datatypes.vehicle_state.ego_state_metadata import (
    imu_se2_to_center_se2,
    rear_axle_se2_to_imu_se2,
)
from py123d.geometry import BoundingBoxSE2, PoseSE2
from py123d.geometry.utils.rotation_utils import normalize_angle
from shapely.geometry import Polygon
from shapely.geometry.base import CAP_STYLE

from py123d_garage.datatypes.trajectory import (
    TrajectorySE2,
)
from py123d_garage.evaluation.navsim.help.observation.pdm_observation import (
    PDMObservation,
)
from py123d_garage.evaluation.navsim.help.proposal.pdm_proposal import (
    PDMProposalManager,
)
from py123d_garage.evaluation.navsim.help.trajectory_grid import TrajectoryGrid
from py123d_garage.evaluation.navsim.help.utils.pdm_constants import (
    DYNAMIC_OBJECT_LABELS,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_enums import (
    LeadingAgentIndex,
    StateIDMIndex,
    StateIndex,
)


@dataclass
class PDMGeneratorState:
    """Dataclass to store the state of the PDMGenerator, for debugging and visualization purposes."""

    state_array: jt.Float64[npt.NDArray[np.float64], "proposals times 11"]
    state_idm_array: jt.Float64[npt.NDArray[np.float64], "proposals times 2"]
    leading_agent_array: jt.Float64[
        npt.NDArray[np.float64],
        "proposals times 3",
    ]

    proposal_manager: PDMProposalManager
    observation: PDMObservation

    initial_ego_state_se2: EgoStateSE2

    driving_corridor_cache: dict[int, Polygon]
    time_point_list: list[Timestamp]

    @property
    def ego_metadata(self) -> EgoStateSE3Metadata:
        """Metadata (dimensions, etc.) of the initial ego state."""
        return self.initial_ego_state_se2.metadata


class PDMGenerator:
    """Class to generate proposals in PDM."""

    def __init__(
        self,
        trajectory_grid: TrajectoryGrid,
        proposal_grid: TrajectoryGrid,
        leading_agent_update_rate: int = 2,
    ):
        """
        Constructor of PDMGenerator.

        Args:
            trajectory_grid: Sampling parameters for final trajectory
            proposal_grid: Sampling parameters for proposals
            leading_agent_update_rate: sample update-rate of leading agent state, defaults to 2
        """
        assert trajectory_grid.interval_us == proposal_grid.interval_us, (
            "PDMGenerator: Proposals and Trajectory must have equal interval length!"
        )

        # trajectory config
        self._trajectory_grid: TrajectoryGrid = trajectory_grid
        self._proposal_grid: TrajectoryGrid = proposal_grid
        _sample_interval = trajectory_grid.interval_s
        assert _sample_interval is not None, "PDMGenerator: interval length must be defined!"
        self._sample_interval: float = _sample_interval

        # generation config
        self._leading_agent_update: int = leading_agent_update_rate

        # lazy loaded
        self._state: PDMGeneratorState | None = None

    def _init_state(
        self,
        initial_ego_state_se2: EgoStateSE2,
        observation: PDMObservation,
        proposal_manager: PDMProposalManager,
    ) -> PDMGeneratorState:
        """
        Re-initializes the unrolling state arrays and caches for a new iteration.

        Args:
            initial_ego_state_se2: ego-vehicle state at t=0
            observation: PDMObservation class
            proposal_manager: PDMProposalManager class

        Returns:
            freshly initialized PDMGeneratorState
        """
        assert initial_ego_state_se2 is not None, "PDMGenerator: initial_ego_state_se2 must be defined!"
        assert observation is not None, "PDMGenerator: observation must be defined!"
        assert proposal_manager is not None, "PDMGenerator: proposal_manager must be defined!"

        # lazy loading
        _proposal_manager = proposal_manager
        _observation = observation
        _initial_ego_state_se2 = initial_ego_state_se2

        # reset proposal state arrays
        _state_array = np.zeros(
            (
                len(_proposal_manager),
                self._trajectory_grid.num_poses + 1,
                len(StateIndex),
            ),
            dtype=np.float64,
        )  # x, y, heading
        _state_idm_array = np.zeros(
            (
                len(_proposal_manager),
                self._trajectory_grid.num_poses + 1,
                2,
            ),
            dtype=np.float64,
        )  # progress, velocity
        _leading_agent_array = np.zeros(
            (
                len(_proposal_manager),
                self._trajectory_grid.num_poses + 1,
                3,
            ),
            dtype=np.float64,
        )  # progress, velocity, rear-length

        # reset caches
        _driving_corridor_cache: dict[int, Polygon] = {}

        initial_time_us = _initial_ego_state_se2.timestamp.time_us
        proposal_num_poses = self._proposal_grid.num_poses
        assert proposal_num_poses is not None, "PDMGenerator: number of proposal poses must be defined!"
        _time_point_list: list[Timestamp] = [Timestamp.from_us(initial_time_us)]
        for time_idx in range(1, proposal_num_poses + 1, 1):
            next_time_point = Timestamp.from_us(
                initial_time_us + int(time_idx * self._sample_interval * 1e6),
            )
            _time_point_list.append(next_time_point)

        return PDMGeneratorState(
            state_array=_state_array,
            state_idm_array=_state_idm_array,
            leading_agent_array=_leading_agent_array,
            proposal_manager=_proposal_manager,
            observation=_observation,
            initial_ego_state_se2=_initial_ego_state_se2,
            driving_corridor_cache=_driving_corridor_cache,
            time_point_list=_time_point_list,
        )

    def generate_proposals(
        self,
        initial_ego_state: EgoStateSE2,
        observation: PDMObservation,
        proposal_manager: PDMProposalManager,
    ) -> jt.Float64[npt.NDArray[np.float64], "proposals times 11"]:
        """
        Generates proposals by unrolling IDM policies for varying paths.

        Saves the proposal states in array representation.

        Args:
            initial_ego_state: state of ego-vehicle at t=0
            observation: PDMObservation class
            proposal_manager: PDMProposalManager class

        Returns:
            unrolled proposal states in array representation
        """
        self._state = self._init_state(
            initial_ego_state,
            observation,
            proposal_manager,
        )

        # unroll proposals per path, to interpolate along batch-dim
        lateral_batch_dict = self._get_lateral_batch_dict()

        for lateral_batch_indices in lateral_batch_dict.values():
            self._initialize_states(lateral_batch_indices)
            for time_idx in range(
                1,
                self._proposal_grid.num_poses + 1,
                1,
            ):
                self._update_leading_agents(lateral_batch_indices, time_idx)
                self._update_idm_states(lateral_batch_indices, time_idx)
                self._update_states_se2(lateral_batch_indices, time_idx)

        return self._state.state_array

    def generate_trajectory(self, proposal_idx: int) -> TrajectorySE2:
        """
        Completes unrolling of the selected proposal to the full trajectory horizon.

        Args:
            proposal_idx: index of best-scored proposal

        Returns:
            the unrolled trajectory as SE2
        """
        assert self._state is not None, "PDMGenerator: call generate_proposals first!"
        assert len(self._state.time_point_list) == self._proposal_grid.num_poses + 1, (
            "PDMGenerator: Proposals must be generated first!"
        )

        lateral_batch_idcs = [proposal_idx]
        current_time_point = self._state.time_point_list[-1].time_us

        for time_idx in range(
            self._proposal_grid.num_poses + 1,
            self._trajectory_grid.num_poses + 1,
            1,
        ):
            current_time_point += int(self._sample_interval * 1e6)
            self._state.time_point_list.append(
                Timestamp.from_us(current_time_point),
            )

            self._update_leading_agents(lateral_batch_idcs, time_idx)
            self._update_idm_states(lateral_batch_idcs, time_idx)
            self._update_states_se2(lateral_batch_idcs, time_idx)

        return TrajectorySE2(
            pose_se2_array=self._state.state_array[
                proposal_idx,
                :,
                StateIndex.STATE_SE2,
            ],
            timestamps_us=np.array(
                [timestamp.time_us for timestamp in self._state.time_point_list],
                dtype=np.int64,
            ),
        )

    def _initialize_states(self, lateral_batch_idcs: list[int]) -> None:
        """
        Initializes all state arrays for ego, IDM, and leading agent at t=0.

        Args:
            lateral_batch_idcs: list of proposal indices, sharing a path.
        """
        assert self._state is not None, "PDMGenerator: call _init_state first!"

        # all initial states are identical for shared lateral_idx
        # thus states are created for lateral_batch_idcs[0] and repeated
        dummy_proposal_idx = lateral_batch_idcs[0]

        ego_position = self._state.initial_ego_state_se2.rear_axle_2d.shapely_point

        ego_progress = self._state.proposal_manager[dummy_proposal_idx].linestring.project(ego_position)
        assert self._state.initial_ego_state_se2.dynamic_state_se2 is not None, (
            "PDMGenerator: initial ego state must have dynamic state information!"
        )
        ego_velocity = self._state.initial_ego_state_se2.dynamic_state_se2.velocity_2d.x

        self._state.state_idm_array[
            lateral_batch_idcs,
            0,
            StateIDMIndex.PROGRESS,
        ] = ego_progress
        self._state.state_idm_array[
            lateral_batch_idcs,
            0,
            StateIDMIndex.VELOCITY,
        ] = ego_velocity

        state_array = self._state.proposal_manager[dummy_proposal_idx].path.interpolate(np.array(ego_progress))
        self._state.state_array[lateral_batch_idcs, 0, StateIndex.STATE_SE2] = state_array

    def _update_states_se2(
        self,
        lateral_batch_idcs: list[int],
        time_idx: int,
    ) -> None:
        """
        Updates the ego state array at the current time-step.

        Args:
            lateral_batch_idcs: list of proposal indices, sharing a path.
            time_idx: index of unrolling iteration (for proposal/trajectory samples)
        """
        assert self._state is not None, "PDMGenerator: call _init_state first!"
        assert time_idx > 0, "PDMGenerator: call _initialize_states first!"
        dummy_proposal_idx = lateral_batch_idcs[0]
        current_progress = self._state.state_idm_array[
            lateral_batch_idcs,
            time_idx,
            StateIDMIndex.PROGRESS,
        ]
        states_se2_array = self._state.proposal_manager[dummy_proposal_idx].path.interpolate(current_progress)
        assert isinstance(states_se2_array, np.ndarray), "PDMGenerator: interpolation must return array representation!"
        assert np.all(np.isfinite(states_se2_array)), (
            f"PDMGenerator: interpolation returned NaN or infinite values for progress {current_progress}!"
        )
        self._state.state_array[
            lateral_batch_idcs,
            time_idx,
            StateIndex.STATE_SE2,
        ] = states_se2_array

    def _update_idm_states(
        self,
        lateral_batch_idcs: list[int],
        time_idx: int,
    ) -> None:
        """
        Updates the IDM state array by propagating the policy for one step.

        Args:
            lateral_batch_idcs: list of proposal indices, sharing a path.
            time_idx: index of unrolling iteration (for proposal/trajectory samples)
        """
        assert self._state is not None, "PDMGenerator: call _init_state first!"
        assert time_idx > 0, "PDMGenerator: call _initialize_states first!"
        longitudinal_idcs = [
            self._state.proposal_manager[proposal_idx].longitudinal_idx for proposal_idx in lateral_batch_idcs
        ]
        next_idm_states = self._state.proposal_manager.longitudinal_policies.propagate(
            self._state.state_idm_array[lateral_batch_idcs, time_idx - 1],
            self._state.leading_agent_array[lateral_batch_idcs, time_idx],
            longitudinal_idcs,
            self._sample_interval,
        )
        assert np.all(np.isfinite(next_idm_states)), (
            f"PDMGenerator: IDM propagation returned NaN or infinite values for time_idx {time_idx}!"
        )
        self._state.state_idm_array[lateral_batch_idcs, time_idx] = next_idm_states

    def _update_leading_agents(
        self,
        lateral_batch_idcs: list[int],
        time_idx: int,
    ) -> None:
        """
        Updates the leading agent state array from agents/obstacles in the driving corridor.

        Args:
            lateral_batch_idcs: list of proposal indices, sharing a path.
            time_idx: index of unrolling iteration (for proposal/trajectory samples)
        """
        assert time_idx > 0, "PDMGenerator: call _initialize_states first!"
        assert self._state is not None, "PDMGenerator: call _init_state first!"

        # update leading agent state at first call or at update rate (runtime)
        update_leading_agent: bool = (time_idx % self._leading_agent_update) == 0

        if not update_leading_agent:
            self._state.leading_agent_array[lateral_batch_idcs, time_idx] = self._state.leading_agent_array[
                lateral_batch_idcs,
                time_idx - 1,
            ]

        else:
            dummy_proposal_idx = lateral_batch_idcs[0]

            leading_agent_array = np.zeros(
                len(LeadingAgentIndex),
                dtype=np.float64,
            )
            intersecting_objects: list[str] = self._get_intersecting_objects(
                lateral_batch_idcs,
                time_idx,
            )

            # collect all leading vehicles ones for all proposals (run-time)
            object_progress_dict: dict[str, float] = {}
            for token in intersecting_objects:
                if token not in self._state.observation.collided_track_ids:
                    object_progress = self._state.proposal_manager[dummy_proposal_idx].linestring.project(
                        self._state.observation[time_idx][token].centroid,
                    )
                    object_progress_dict[token] = object_progress

            # select leading agent for each proposal individually
            for proposal_idx in lateral_batch_idcs:
                current_ego_progress = self._state.state_idm_array[
                    proposal_idx,
                    time_idx - 1,
                    StateIDMIndex.PROGRESS,
                ]

                # filter all objects ahead
                agents_ahead: dict[str, float] = {
                    agent: progress
                    for agent, progress in object_progress_dict.items()
                    if progress > current_ego_progress
                }

                if len(agents_ahead) > 0:  # red light, object or agent ahead
                    current_state_se2 = PoseSE2.from_array(
                        self._state.state_array[
                            proposal_idx,
                            time_idx - 1,
                            StateIndex.STATE_SE2,
                        ],
                    )
                    imu_se2 = rear_axle_se2_to_imu_se2(
                        current_state_se2,
                        self._state.ego_metadata,
                    )
                    center_se2 = imu_se2_to_center_se2(
                        imu_se2,
                        self._state.ego_metadata,
                    )

                    ego_polygon = BoundingBoxSE2(
                        center_se2=center_se2,
                        length=self._state.ego_metadata.length,
                        width=self._state.ego_metadata.width,
                    ).shapely_polygon

                    relative_distances = [
                        ego_polygon.distance(
                            self._state.observation[time_idx][agent],
                        )
                        for agent in agents_ahead
                    ]

                    argmin = np.argmin(relative_distances)
                    nearest_agent = list(agents_ahead.keys())[argmin]

                    # add rel. distance for red light, object or agent
                    relative_distance = current_ego_progress + relative_distances[argmin]
                    leading_agent_array[LeadingAgentIndex.PROGRESS] = relative_distance

                    # calculate projected velocity if not red light
                    if self._state.observation.red_light_token not in nearest_agent:
                        leading_agent_array[LeadingAgentIndex.VELOCITY] = self._get_leading_agent_velocity(
                            ego_yaw=current_state_se2.yaw,
                            agent=self._state.observation.unique_objects[nearest_agent],
                        )

                else:  # nothing ahead, free driving
                    path_length = self._state.proposal_manager[proposal_idx].linestring.length
                    path_rear = self._state.initial_ego_state_se2.metadata.length / 2

                    leading_agent_array[LeadingAgentIndex.PROGRESS] = path_length
                    leading_agent_array[LeadingAgentIndex.LENGTH_REAR] = path_rear

                self._state.leading_agent_array[proposal_idx, time_idx] = leading_agent_array

    @staticmethod
    def _get_leading_agent_velocity(
        ego_yaw: float,
        agent: BoxDetectionSE2,
    ) -> float:
        """
        Calculates velocity of leading vehicle projected onto ego's heading.

        Args:
            ego_yaw: heading angle [rad]
            agent: the leading object

        Returns:
            projected velocity [m/s]
        """
        if (
            isinstance(agent, BoxDetectionSE2) and agent.attributes.default_label in DYNAMIC_OBJECT_LABELS
        ):  # dynamic object
            relative_heading = normalize_angle(agent.center_se2.yaw - ego_yaw)
            agent_global_velocity = agent.velocity_2d
            assert agent_global_velocity is not None, "PDMGenerator: dynamic object has no velocity information!"
            projected_velocity = agent_global_velocity.magnitude * np.cos(
                relative_heading,
            )
        else:  # static object
            projected_velocity = 0.0
        return float(projected_velocity)

    def _get_intersecting_objects(
        self,
        lateral_batch_idcs: list[int],
        time_idx: int,
    ) -> list[str]:
        """
        Returns all objects intersecting the proposals' driving corridor at a time-step.

        Args:
            lateral_batch_idcs: list of proposal indices, sharing a path
            time_idx: index of unrolling iteration (for proposal/trajectory samples)

        Returns:
            list of object tokens
        """
        assert self._state is not None, "PDMGenerator: call _init_state first!"
        dummy_proposal_idx = lateral_batch_idcs[0]
        driving_corridor: Polygon = self._get_driving_corridor(
            dummy_proposal_idx,
        )
        # The observation's occupancy maps are keyed by string track tokens.
        return cast(
            list[str],
            self._state.observation[time_idx].intersects(driving_corridor),
        )

    def _get_driving_corridor(self, proposal_idx: int) -> Polygon:
        """
        Creates and caches the driving corridor of the ego-vehicle for each proposal path.

        Args:
            proposal_idx: index of a proposal

        Returns:
            polygon spanning the max trajectory distance, buffered by ego's width
        """
        assert self._state is not None, "PDMGenerator: call _init_state first!"
        lateral_idx = self._state.proposal_manager[proposal_idx].lateral_idx

        if lateral_idx not in self._state.driving_corridor_cache:
            ego_distance = self._state.state_idm_array[
                proposal_idx,
                0,
                StateIDMIndex.PROGRESS,
            ]
            trajectory_distance = (
                ego_distance
                + abs(self._state.proposal_manager.max_target_velocity)
                * self._trajectory_grid.num_poses
                * self._sample_interval
            )
            linestring_ahead = self._state.proposal_manager[proposal_idx].path.subline(
                ego_distance,
                trajectory_distance,
            )
            expanded_path = linestring_ahead.linestring.buffer(
                self._state.ego_metadata.width / 2,
                cap_style=CAP_STYLE.square,
            )
            self._state.driving_corridor_cache[lateral_idx] = expanded_path

        return self._state.driving_corridor_cache[lateral_idx]

    def _get_lateral_batch_dict(self) -> dict[int, list[int]]:
        """
        Groups proposal indices by their shared lateral path.

        Returns:
            dictionary mapping lateral index to its proposal indices
        """
        assert self._state is not None, "PDMGenerator: call _init_state first!"
        lateral_batch_dict: dict[int, list[int]] = {}

        for proposal_idx in range(len(self._state.proposal_manager)):
            lateral_idx = self._state.proposal_manager[proposal_idx].lateral_idx

            if lateral_idx not in lateral_batch_dict:
                lateral_batch_dict[lateral_idx] = [proposal_idx]
            else:
                lateral_batch_dict[lateral_idx].append(proposal_idx)

        return lateral_batch_dict
