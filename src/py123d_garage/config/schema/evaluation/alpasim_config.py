# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import override

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig
from py123d_garage.datatypes.numerics import NonNegativeFloat, PositiveFloat, PositiveInt


@dataclass
class EgoVehicleConfig:
    """Ego vehicle geometry served to the AlpaSim driver."""

    # Vehicle name written into the ego metadata.
    vehicle_name: str = "alpasim"
    # Bounding-box width of the ego vehicle.
    width_m: float = 2.297
    # Bounding-box length of the ego vehicle.
    length_m: float = 5.176
    # Bounding-box height of the ego vehicle.
    height_m: float = 1.777
    # Distance between the front and rear axles.
    wheel_base_m: float = 3.089
    # Longitudinal offset from the rear axle to the box center.
    rear_axle_to_center_longitudinal_m: float = 1.461


@dataclass
class AlpasimBenchmarkConfig(AbstractBenchmarkConfig):
    """Config of the AlpaSim gRPC driver service."""

    # -- Config objects --

    # The policy under evaluation; evaluation_checkpoint_files select its weights.
    policy_config: EvaluationPolicyConfig = field(default_factory=EvaluationPolicyConfig)
    # Ego geometry the driver builds its metadata from.
    ego_vehicle_config: EgoVehicleConfig = field(default_factory=EgoVehicleConfig)

    # -- Atomic settings --

    # Git hash of the code baked into the submission image.
    git_hash: str = "unknown"

    # Inference device of the driver process.
    device: str = "cuda"
    # Bind address of the gRPC driver service.
    host: str = "0.0.0.0"
    # Port of the gRPC driver service.
    port: int = 6789
    # gRPC server worker threads.
    max_workers: int = 8
    # Writes a drive video every n-th plan of every session into <run dir>/views; 0 disables the recording.
    visualization_interval: int = 0
    # Writes every plan's ego state and model outputs per session to <run dir>/plans/<scene>_<session>.npz.
    plan_log: bool = False
    # Sim time between inference runs, the stored plan serving the calls between; 0 means the first trajectory offset.
    replan_interval_us: int = 500_000
    # Spacing the route polyline is resampled at.
    route_resolution_m: float = 0.1
    # Merges each served route into the route of earlier plans, so the near section the runtime withholds stays known.
    route_memory: bool = False
    # Extends the route straight along its final heading when it ends before the farthest target point.
    extrapolate_route: bool = False
    # Hood from common/sensor/assets/ego_hoods pasted onto the front camera; None leaves the rendered frame as it is.
    ego_hood: str | None = None
    # Hood the renderer already draws on the front camera, blurred away under the pasted one; None when it renders none.
    rendered_ego_hood: str | None = "hyperion_8_1"

    # -- The AlpaSim protocol, hard-coded --

    @property
    @override
    def required_trajectory_horizon_us(self) -> PositiveInt:
        """Covers the 500 ms replan interval plus the MPC's 2 s reference lookahead."""
        return 2_500_000

    @property
    @override
    def min_served_target_point_distance_m(self) -> NonNegativeFloat:
        """The challenge wizard sets route_start_offset_m=40, so the served route starts 40 m ahead of the ego."""
        return 40.0

    @property
    @override
    def max_served_target_point_distance_m(self) -> PositiveFloat:
        """The route generator looks 80 m ahead along the path."""
        return 80.0

    @property
    @override
    def served_ego_state_interval_us(self) -> PositiveInt:
        """The simulator drives at 10 Hz ticks."""
        return 100_000

    @property
    @override
    def served_camera_interval_us(self) -> PositiveInt:
        """Camera frames arrive once per drive tick."""
        return 100_000

    @property
    @override
    def served_lidar_interval_us(self) -> PositiveInt | None:
        """Alpasim does not have lidar."""
        return None
