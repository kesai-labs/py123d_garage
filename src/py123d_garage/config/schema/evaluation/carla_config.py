from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING
from typing_extensions import override

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig
from py123d_garage.datatypes.numerics import PositiveInt


@dataclass
class CarlaBenchmarkConfig(AbstractBenchmarkConfig):
    # -- Config objects --

    # The policy under evaluation; evaluation_checkpoint_file selects its weights.
    policy_config: EvaluationPolicyConfig = field(default_factory=EvaluationPolicyConfig)

    # -- Atomic settings --

    # Leaderboard routes XML the evaluator drives.
    routes_file: str = MISSING
    # Leaderboard flavor under lib/carla/benchmark the routes belong to: standard or bench2drive.
    benchmark: str = "standard"
    # Dataset name written into the live scene's log metadata.
    dataset: str = "carla"
    # Town name written into the live scene's log metadata.
    location: str = ""
    # Past ticks the live scene keeps for the policy's history.
    num_history_iterations: int = 4
    # Smooths the GNSS and compass readings into the ego pose.
    use_kalman_filter: bool = True
    # Writes a drive video every n-th tick; 0 disables the recording.
    visualization_interval: int = 0
    # TCP port of the CARLA server.
    port: int = 2000
    # TCP port of the CARLA traffic manager.
    traffic_manager_port: int = 8000
    # Seed of the traffic manager's random traffic behavior.
    traffic_manager_seed: int = 0
    # How often every route in the routes file is driven.
    repetitions: int = 1
    # CARLA client timeout in seconds.
    timeout: float = 300.0
    # Continues an earlier run from the results.json in the output dir.
    resume: bool = False
    # Leaderboard debug verbosity; 0 prints no debug output.
    debug: int = 0

    # -- The CARLA leaderboard protocol, hard-coded --

    @property
    @override
    def required_trajectory_horizon_us(self) -> PositiveInt:
        """Window the agent's controller tracks waypoints over; replanning happens every tick."""
        return 2_000_000

    @property
    @override
    def served_ego_state_interval_us(self) -> PositiveInt:
        """The leaderboard ticks at CARLA_FPS."""
        return 50_000

    @property
    @override
    def served_camera_interval_us(self) -> PositiveInt:
        """The leaderboard ticks at CARLA_FPS."""
        return 50_000

    @property
    @override
    def served_lidar_interval_us(self) -> PositiveInt:
        """The leaderboard ticks at CARLA_FPS."""
        return 50_000
