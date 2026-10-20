# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from typing import cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api import SceneAPI

from py123d_garage.datatypes.trajectory import (
    TrajectorySE2,
    TrajectoryXY,
)
from py123d_garage.evaluation.navsim.help.observation.pdm_observation import (
    PDMObservation,
)
from py123d_garage.evaluation.navsim.help.pdm_agent import PDMAgent
from py123d_garage.evaluation.navsim.help.scoring.pdm_scorer import PDMScorer
from py123d_garage.evaluation.navsim.help.simulation.pdm_simulator import (
    PDMSimulator,
)
from py123d_garage.evaluation.navsim.help.trajectory_grid import TrajectoryGrid
from py123d_garage.evaluation.navsim.help.trajectory_utils import (
    resample_trajectory_se2,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_enums import StateIndex


class PDMMetric:
    """Scores an agent trajectory with the PDM-Closed closed-loop simulation and scorer."""

    def __init__(self, route_correction: bool = True) -> None:
        """
        Sets up the metric, which scores 4 s at 0.1 s intervals.

        Args:
            route_correction: Whether PDM-Closed corrects its route on the first iteration.
        """
        self._score_trajectory_grid = TrajectoryGrid.from_horizon_and_interval(
            horizon_us=4_000_000,
            interval_us=100_000,
        )
        self._route_correction = route_correction

    def compute_metric(
        self,
        scene_api: SceneAPI,
        **kwargs: TrajectoryXY | TrajectorySE2,
    ) -> dict[str, float]:
        """
        Scores the agent trajectory on a scene against the PDM-Closed reference.

        Args:
            scene_api: Scene to score on.
            **kwargs: Must hold agent_trajectory, the ego-relative trajectory to score.

        Returns:
            The PDM sub-scores and the final score of the agent trajectory.
        """
        assert "agent_trajectory" in kwargs, "Missing required argument: agent_trajectory"
        agent_trajectory = kwargs["agent_trajectory"]
        assert isinstance(agent_trajectory, TrajectoryXY | TrajectorySE2), (
            "Argument 'agent_trajectory' must be a TrajectoryXY or a TrajectorySE2"
        )

        pdm_agent = PDMAgent(route_correction=self._route_correction)
        map_api = scene_api.get_map_api()
        assert map_api is not None, "MapAPI not found in SceneAPI."
        pdm_agent.initialize()

        pdm_trajectory = pdm_agent.compute_trajectory(scene_api)
        assert isinstance(pdm_trajectory, TrajectorySE2), "PDM-Closed trajectory must be of type TrajectorySE2."

        _ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
        assert _ego_state_se3 is not None, "Initial ego state SE3 not found in SceneAPI."
        initial_ego_state_se2 = _ego_state_se3.ego_state_se2

        # PDM-Closed plans in the absolute frame, the agent trajectory is ego-relative.
        resampled_pdm_trajectory = resample_trajectory_se2(
            trajectory=pdm_trajectory,
            trajectory_grid=self._score_trajectory_grid,
            initial_ego_state_se2=initial_ego_state_se2,
            convert_to_absolute=False,
        )
        resampled_agent_trajectory = resample_trajectory_se2(
            trajectory=agent_trajectory.to_se2(),
            trajectory_grid=self._score_trajectory_grid,
            initial_ego_state_se2=initial_ego_state_se2,
            convert_to_absolute=True,
        )
        states = _convert_trajectory_to_state_array(
            [
                resampled_pdm_trajectory,
                resampled_agent_trajectory,
            ],
        )

        simulated_states = PDMSimulator(
            self._score_trajectory_grid,
        ).simulate_proposals(
            states=states,
            initial_ego_state=initial_ego_state_se2,
        )

        # TODO@DanielDauner: This needs a cleaner solution, needs to be refactored together with PDMScorer.
        assert pdm_agent._drivable_area_map is not None, "PDM-Closed planner drivable area map is not initialized."
        assert pdm_agent._route_lane_group_dict is not None, "PDM-Closed planner route lane groups are not initialized."
        assert pdm_agent._centerline is not None, "PDM-Closed planner centerline is not initialized."
        drivable_area_map = pdm_agent._drivable_area_map
        route_lane_ids = list(pdm_agent._route_lane_group_dict.keys())
        centerline = pdm_agent._centerline

        log_replay_observation = PDMObservation(
            trajectory_grid=TrajectoryGrid.from_horizon_and_interval(
                horizon_us=8_000_000,
                interval_us=100_000,
            ),
            proposal_grid=TrajectoryGrid.from_horizon_and_interval(
                horizon_us=4_000_000,
                interval_us=100_000,
            ),
            map_radius=100.0,
            observation_sample_res=1,
            extend_observation_for_ttc=True,
        )
        log_replay_observation.update_replay(scene_api)

        pdm_scores = PDMScorer(
            proposal_grid=self._score_trajectory_grid,
        ).score_proposals(
            states=simulated_states,
            observation=log_replay_observation,
            centerline=centerline,
            route_lane_ids=route_lane_ids,
            drivable_area_map=drivable_area_map,
            ego_metadata=initial_ego_state_se2.metadata,
        )

        return cast(
            dict[str, float],
            pdm_scores[1].iloc[0].to_dict(),
        )


def _convert_trajectory_to_state_array(
    trajectories: list[TrajectorySE2],
) -> jt.Float64[npt.NDArray[np.float64], "trajectories poses 11"]:
    """
    Stacks trajectories into the state array the PDM modules take.

    Args:
        trajectories: Trajectories with the same number of poses.

    Returns:
        The state array with only the pose entries filled.
    """
    # TODO@DanielDauner: This is a temporary solution, we should refactor the PDM modules to work with TrajectorySE2 directly.
    num_poses_ = trajectories[0].pose_se2_array.shape[0]

    _state_array = np.zeros(
        (
            len(trajectories),
            num_poses_,
            len(StateIndex),
        ),
        dtype=np.float64,
    )

    for traj_idx, trajectory in enumerate(trajectories):
        assert trajectory.pose_se2_array.shape[0] == num_poses_, "All trajectories must have the same number of poses."
        _state_array[traj_idx, :num_poses_, StateIndex.STATE_SE2] = trajectory.pose_se2_array

    return _state_array
