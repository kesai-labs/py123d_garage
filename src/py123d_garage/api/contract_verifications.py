from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from py123d.api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.api.abstract_offline_data_source_config import AbstractOfflineDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.datatypes.numerics import NonNegativeFloat, PositiveFloat, PositiveInt

LOG = logging.getLogger(__name__)

_MIN_TIMESTAMPS = 4
_PERIOD_TOLERANCE = 0.2
_DROP_WARNING_FRACTION = 0.05


def verify_policy_against_scene_filter(
    policy_config: AbstractPolicyConfig,
    scene_filter: SceneFilter,
) -> None:
    """
    Refuses a scene filter whose guarantees cannot serve the policy's requirements.

    Args:
        policy_config: the policy's declared requirements.
        scene_filter: the filter about to select scenes for the policy.

    Raises:
        ValueError: on the first guarantee that falls short.
    """
    guaranteed_cameras = {
        modality.removeprefix("camera.").partition("@")[0]
        for modality in scene_filter.required_scene_modalities or []
        if modality.startswith("camera.")
    }
    if (
        policy_config.required_cameras
        and guaranteed_cameras
        and not any(
            {camera.name.lower() for camera in cameras} <= guaranteed_cameras
            for cameras in policy_config.required_cameras.values()
        )
    ):
        raise ValueError(
            f"the scene filter guarantees cameras {sorted(guaranteed_cameras)}, which cover "
            f"no dataset of the policy's required_cameras {policy_config.required_cameras}",
        )
    guaranteed_history_us = round((scene_filter.history_duration_s or 0.0) * 1e6)
    if guaranteed_history_us < policy_config.required_history_duration_us:
        raise ValueError(
            f"filter history_duration_s={scene_filter.history_duration_s} cannot span "
            f"the {policy_config.required_history_duration_us} µs of history the policy "
            "consumes: past frames would be clipped at the scene boundary.",
        )
    guaranteed_future_us = round((scene_filter.future_duration_s or 0.0) * 1e6)
    if guaranteed_future_us < policy_config.trajectory_horizon_us:
        raise ValueError(
            f"filter future_duration_s={scene_filter.future_duration_s} cannot span "
            f"the {policy_config.trajectory_horizon_us} µs trajectory horizon: trajectory "
            "labels would be clipped at the scene boundary.",
        )
    farthest_target_point_m = max(policy_config.required_target_point_distances_m)
    if scene_filter.min_remaining_route_m is None or scene_filter.min_remaining_route_m < farthest_target_point_m:
        raise ValueError(
            f"filter min_remaining_route_m={scene_filter.min_remaining_route_m} cannot "
            f"reach the farthest target point at {farthest_target_point_m} m: scenes "
            "whose route falls short would fail per sample.",
        )


def verify_policy_against_offline_data_source(
    policy_config: AbstractPolicyConfig,
    offline_data_source_config: AbstractOfflineDataSourceConfig,
) -> None:
    """
    Fails fast when a source cannot serve what the policy consumes.

    Args:
        policy_config: the policy's declared requirements.
        offline_data_source_config: the data source's declared spacings and limits.

    Raises:
        ValueError: if a required modality is absent, its spacing cannot serve
            the required interval as a whole number of frames, or a target
            point falls outside the served range.
    """
    provider = f"the source at '{offline_data_source_config.data_root}'"
    _verify_policy_against_provider_sensor_intervals(
        policy_config,
        served_ego_state_interval_us=offline_data_source_config.served_ego_state_interval_us,
        served_camera_interval_us=offline_data_source_config.served_camera_interval_us,
        served_lidar_interval_us=offline_data_source_config.served_lidar_interval_us,
        provider=provider,
    )
    _verify_policy_against_provider_target_points(
        policy_config,
        min_distance_m=offline_data_source_config.min_served_target_point_distance_m,
        max_distance_m=offline_data_source_config.max_served_target_point_distance_m,
        provider=provider,
    )


def verify_policy_against_benchmark(
    policy_config: AbstractPolicyConfig,
    benchmark_config: AbstractBenchmarkConfig,
) -> None:
    """
    Verify that a benchmark's protocol and data sources can meet a policy's requirements.

    Args:
        policy_config: the policy's declared requirements and output shape.
        benchmark_config: the benchmark's declared protocol and data sources.

    Raises:
        ValueError: on the first declaration the benchmark or one of its
            data sources cannot meet.
    """
    if (
        benchmark_config.required_trajectory_horizon_us is not None
        and policy_config.trajectory_horizon_us < benchmark_config.required_trajectory_horizon_us
    ):
        raise ValueError(
            f"the policy plans {policy_config.trajectory_horizon_us} µs ahead but this "
            f"benchmark scores a {benchmark_config.required_trajectory_horizon_us} µs window.",
        )
    if (
        benchmark_config.max_history_duration_us is not None
        and policy_config.required_history_duration_us > benchmark_config.max_history_duration_us
    ):
        raise ValueError(
            f"the policy's features reach {policy_config.required_history_duration_us} µs back but "
            f"this benchmark's protocol serves at most {benchmark_config.max_history_duration_us} µs of past.",
        )
    _verify_policy_against_provider_target_points(
        policy_config,
        min_distance_m=benchmark_config.min_served_target_point_distance_m,
        max_distance_m=benchmark_config.max_served_target_point_distance_m,
        provider=type(benchmark_config).__name__,
    )
    declared_intervals = (
        benchmark_config.served_ego_state_interval_us,
        benchmark_config.served_camera_interval_us,
        benchmark_config.served_lidar_interval_us,
    )
    if any(interval is not None for interval in declared_intervals):
        _verify_policy_against_provider_sensor_intervals(
            policy_config,
            served_ego_state_interval_us=benchmark_config.served_ego_state_interval_us,
            served_camera_interval_us=benchmark_config.served_camera_interval_us,
            served_lidar_interval_us=benchmark_config.served_lidar_interval_us,
            provider=type(benchmark_config).__name__,
        )
    for offline_data_source_config in benchmark_config.benchmark_offline_data_sources.values():
        verify_policy_against_offline_data_source(policy_config, offline_data_source_config)


def verify_offline_data_source_scenes_declared_intervals(
    offline_data_source_config: AbstractOfflineDataSourceConfig,
    scene_api: SceneAPI,
) -> None:
    """
    Fails fast when a scene's recorded timing contradicts a source's declared spacings.

    Args:
        offline_data_source_config: the data source whose declared spacings to verify.
        scene_api: any scene built from the source; one per source suffices.

    Raises:
        ValueError: if a stream's measured period is off its declaration by
            more than 20%.
    """
    provider = f"the source at '{offline_data_source_config.data_root}'"
    _verify_stream_records_declared_interval(
        [t.time_us for t in scene_api.get_all_ego_state_se3_timestamps(include_history=True)],
        interval_us=offline_data_source_config.served_ego_state_interval_us,
        provider=provider,
        stream="ego states",
    )
    for camera_id in scene_api.available_camera_ids:
        _verify_stream_records_declared_interval(
            [t.time_us for t in scene_api.get_all_camera_timestamps(camera_id, include_history=True)],
            interval_us=offline_data_source_config.served_camera_interval_us,
            provider=provider,
            stream=f"camera {camera_id.name}",
        )
    if offline_data_source_config.served_lidar_interval_us is not None:
        for lidar_id in scene_api.available_lidar_ids:
            _verify_stream_records_declared_interval(
                [t.time_us for t in scene_api.get_all_lidar_timestamps(lidar_id, include_history=True)],
                interval_us=offline_data_source_config.served_lidar_interval_us,
                provider=provider,
                stream=f"lidar {lidar_id.name}",
            )


def _verify_policy_against_provider_sensor_intervals(
    policy_config: AbstractPolicyConfig,
    *,
    served_ego_state_interval_us: PositiveInt | None,
    served_camera_interval_us: PositiveInt | None,
    served_lidar_interval_us: PositiveInt | None,
    provider: str,
) -> None:
    """
    Fails fast when a provider cannot serve frames at the intervals the policy consumes.

    Args:
        policy_config: the policy's declared requirements.
        served_ego_state_interval_us: the provider ego-state spacing; None = modality absent.
        served_camera_interval_us: the provider camera spacing; None = modality absent.
        served_lidar_interval_us: the provider lidar spacing; None = modality absent.
        provider: names the serving side in error messages.

    Raises:
        ValueError: if a required modality is absent or its spacing cannot serve
            the required interval as a whole number of frames.
    """
    requirements: dict[str, tuple[PositiveInt | None, PositiveInt | None]] = {
        "ego_state": (policy_config.required_past_ego_state_interval_us, served_ego_state_interval_us),
        "camera": (policy_config.required_past_camera_interval_us, served_camera_interval_us),
        "lidar": (policy_config.required_past_lidar_interval_us, served_lidar_interval_us),
    }
    for modality, (required_us, served_us) in requirements.items():
        if required_us is None:
            continue
        if served_us is None:
            raise ValueError(
                f"the policy consumes {modality} frames every {required_us} µs but {provider} serves no {modality}.",
            )
        if required_us % served_us:
            raise ValueError(
                f"the policy consumes {modality} frames every {required_us} µs but {provider} "
                f"serves {modality} every {served_us} µs; the interval must be a whole "
                f"number of frames.",
            )


def _verify_policy_against_provider_target_points(
    policy_config: AbstractPolicyConfig,
    *,
    min_distance_m: NonNegativeFloat | None,
    max_distance_m: PositiveFloat | None,
    provider: str,
) -> None:
    """
    Fails fast when a provider cannot place the target points the policy is conditioned on.

    Args:
        policy_config: the policy's declared requirements.
        min_distance_m: nearest distance the provider can place a target point at; None = no limit.
        max_distance_m: farthest distance the provider can place a target point at; None = no limit.
        provider: names the serving side in error messages.

    Raises:
        ValueError: if a distance falls outside the limits.
    """
    distances_m = policy_config.required_target_point_distances_m
    if min_distance_m is not None and min(distances_m) < min_distance_m:
        raise ValueError(
            f"the policy needs a target point at {min(distances_m)} m but {provider} "
            f"serves none closer than {min_distance_m} m.",
        )
    if max_distance_m is not None and max(distances_m) > max_distance_m:
        raise ValueError(
            f"the policy needs a target point at {max(distances_m)} m but {provider} "
            f"serves none farther than {max_distance_m} m.",
        )


def _verify_stream_records_declared_interval(
    timestamps_us: Sequence[int],
    *,
    interval_us: PositiveInt,
    provider: str,
    stream: str,
) -> None:
    """
    Fails fast when recorded timestamps contradict a declared stream interval.

    Dropped frames only ever lengthen gaps, so the period is estimated from the
    short end of the delta distribution and stays correct up to 75% drops;
    the holes themselves are counted against it and reported as a warning.

    Args:
        timestamps_us: the stream's recorded timestamps in microseconds, ordered; a
            clip-relative clock may lead zero, so negative values are accepted;
            fewer than four are too short to judge and pass silently.
        interval_us: the declared spacing to verify.
        provider: names the serving side in messages.
        stream: names the stream in messages, e.g. "camera PCAM_F0".

    Raises:
        ValueError: if the measured period is off the declared one by more than 20%.
    """
    if len(timestamps_us) < _MIN_TIMESTAMPS:
        return
    deltas_us = np.diff(np.asarray(timestamps_us, dtype=np.int64))
    deltas_us = deltas_us[deltas_us > 0]
    if len(deltas_us) < _MIN_TIMESTAMPS - 1:
        return
    period_us = float(np.percentile(deltas_us, 25))
    if abs(period_us / interval_us - 1.0) > _PERIOD_TOLERANCE:
        raise ValueError(
            f"{provider} declares {stream} every {interval_us} µs but a scene records them ~{period_us:.0f} µs apart.",
        )
    num_missing = int(np.sum(np.round(deltas_us / period_us)) - len(deltas_us))
    num_expected = len(deltas_us) + num_missing
    if num_missing / num_expected > _DROP_WARNING_FRACTION:
        LOG.warning(
            f"{provider} drops {num_missing} of {num_expected} {stream} frames in a scene.",
        )
