from __future__ import annotations

import logging
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from py123d.api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.datatypes import CameraID, Timestamp
from typing_extensions import override

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.api.abstract_offline_data_source_config import AbstractOfflineDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.api.contract_verifications import (
    _verify_stream_records_declared_interval,
    verify_offline_data_source_scenes_declared_intervals,
    verify_policy_against_benchmark,
    verify_policy_against_offline_data_source,
    verify_policy_against_scene_filter,
)


@dataclass
class _Policy(AbstractPolicyConfig):
    trajectory_horizon_us: int = 4_000_000
    trajectory_interval_us: int = 500_000
    required_target_point_distances_m: list[float] = field(
        default_factory=lambda: [45.0],
    )
    history_duration_us: int = 0
    past_ego_state_us: int | None = None
    past_camera_us: int | None = None
    past_lidar_us: int | None = None
    cameras: dict[str, list[CameraID]] = field(default_factory=dict)

    @property
    @override
    def required_cameras(self) -> dict[str, list[CameraID]]:
        return self.cameras

    @property
    @override
    def required_history_duration_us(self) -> int:
        return self.history_duration_us

    @property
    @override
    def required_past_ego_state_interval_us(self) -> int | None:
        return self.past_ego_state_us

    @property
    @override
    def required_past_camera_interval_us(self) -> int | None:
        return self.past_camera_us

    @property
    @override
    def required_past_lidar_interval_us(self) -> int | None:
        return self.past_lidar_us


@dataclass
class _LimitedBenchmarkConfig(AbstractBenchmarkConfig):
    limits: tuple[float | None, float | None] = (None, None)
    required_horizon_us: int | None = None
    max_history_us: int | None = None

    @property
    @override
    def required_trajectory_horizon_us(self) -> int | None:
        return self.required_horizon_us

    @property
    @override
    def max_history_duration_us(self) -> int | None:
        return self.max_history_us

    @property
    @override
    def min_served_target_point_distance_m(self) -> float | None:
        return self.limits[0]

    @property
    @override
    def max_served_target_point_distance_m(self) -> float | None:
        return self.limits[1]


def _benchmark(
    min_distance_m: float | None = None,
    max_distance_m: float | None = None,
) -> AbstractBenchmarkConfig:
    return _LimitedBenchmarkConfig(
        limits=(min_distance_m, max_distance_m),
    )


def _policy(distances_m: list[float], **kwargs: object) -> _Policy:
    return _Policy(required_target_point_distances_m=distances_m, **kwargs)  # pyright: ignore[reportArgumentType]


def _source(
    camera_us: int = 100_000,
    lidar_us: int | None = 100_000,
) -> AbstractOfflineDataSourceConfig:
    return AbstractOfflineDataSourceConfig(
        data_root="/data/123D",
        served_ego_state_interval_us=100_000,
        served_camera_interval_us=camera_us,
        served_lidar_interval_us=lidar_us,
    )


# -- verify_policy_against_scene_filter --


def test_sufficient_filter_passes() -> None:
    verify_policy_against_scene_filter(
        _Policy(history_duration_us=2_000_000),
        SceneFilter(future_duration_s=4.0, min_remaining_route_m=45.0, history_duration_s=2.0),
    )


def test_short_future_raises() -> None:
    with pytest.raises(ValueError, match="future_duration_s"):
        verify_policy_against_scene_filter(
            _Policy(history_duration_us=2_000_000),
            SceneFilter(future_duration_s=2.0, min_remaining_route_m=45.0, history_duration_s=2.0),
        )


def test_short_history_raises() -> None:
    with pytest.raises(ValueError, match="history_duration_s"):
        verify_policy_against_scene_filter(
            _Policy(history_duration_us=2_000_000),
            SceneFilter(future_duration_s=4.0, min_remaining_route_m=45.0, history_duration_s=1.0),
        )


def test_absent_history_raises() -> None:
    with pytest.raises(ValueError, match="history_duration_s"):
        verify_policy_against_scene_filter(
            _Policy(history_duration_us=2_000_000),
            SceneFilter(future_duration_s=4.0, min_remaining_route_m=45.0),
        )


def test_short_route_raises() -> None:
    with pytest.raises(ValueError, match="min_remaining_route_m"):
        verify_policy_against_scene_filter(
            _Policy(history_duration_us=2_000_000),
            SceneFilter(future_duration_s=4.0, min_remaining_route_m=20.0, history_duration_s=2.0),
        )


def test_covered_cameras_pass() -> None:
    verify_policy_against_scene_filter(
        _Policy(cameras={"nuplan": [CameraID.PCAM_F0]}),
        SceneFilter(
            future_duration_s=4.0,
            min_remaining_route_m=45.0,
            required_scene_modalities=["camera.pcam_f0@initial", "camera.pcam_l0@initial"],
        ),
    )


def test_uncovered_cameras_raise() -> None:
    with pytest.raises(ValueError, match="cover no dataset"):
        verify_policy_against_scene_filter(
            _Policy(cameras={"nuplan": [CameraID.PCAM_F0, CameraID.PCAM_L0]}),
            SceneFilter(
                future_duration_s=4.0,
                min_remaining_route_m=45.0,
                required_scene_modalities=["camera.pcam_f0@initial"],
            ),
        )


def test_cameraless_policy_ignores_guaranteed_cameras() -> None:
    verify_policy_against_scene_filter(
        _Policy(),
        SceneFilter(
            future_duration_s=4.0,
            min_remaining_route_m=45.0,
            required_scene_modalities=["camera.pcam_f0@initial"],
        ),
    )


def test_single_frame_policy_needs_no_history() -> None:
    verify_policy_against_scene_filter(
        _Policy(history_duration_us=0),
        SceneFilter(future_duration_s=4.0, min_remaining_route_m=45.0),
    )


# -- verify_policy_against_benchmark --


def test_unconstrained_accepts_any_policy() -> None:
    verify_policy_against_benchmark(
        _policy([4.0, 45.0, 200.0], trajectory_horizon_us=500_000, history_duration_us=100_000_000),
        _benchmark(),
    )


def test_distance_over_max_limit_raises() -> None:
    with pytest.raises(ValueError, match="farther"):
        verify_policy_against_benchmark(
            _policy([45.0, 80.0]),
            _benchmark(max_distance_m=50.0),
        )


def test_distance_under_min_limit_raises() -> None:
    with pytest.raises(ValueError, match="closer"):
        verify_policy_against_benchmark(
            _policy([4.0, 45.0]),
            _benchmark(min_distance_m=40.0),
        )


def test_short_trajectory_horizon_raises() -> None:
    """A policy planning less than the scored window must be refused, not padded."""
    scored = _LimitedBenchmarkConfig(
        required_horizon_us=4_000_000,
    )
    verify_policy_against_benchmark(_policy([45.0], trajectory_horizon_us=4_000_000), scored)
    with pytest.raises(ValueError, match=r"scores a 4000000 µs window"):
        verify_policy_against_benchmark(_policy([45.0], trajectory_horizon_us=2_000_000), scored)


def test_history_over_the_protocol_cap_raises() -> None:
    """A policy reading more past than the benchmark's protocol serves must be refused."""
    capped = _LimitedBenchmarkConfig(max_history_us=1_500_000)
    verify_policy_against_benchmark(_policy([45.0], history_duration_us=1_500_000), capped)
    with pytest.raises(ValueError, match="at most 1500000"):
        verify_policy_against_benchmark(_policy([45.0], history_duration_us=3_500_000), capped)


def test_within_limits_passes() -> None:
    verify_policy_against_benchmark(
        _policy([45.0, 80.0]),
        _benchmark(min_distance_m=40.0, max_distance_m=80.0),
    )


def test_benchmark_checks_its_sources() -> None:
    benchmark = _LimitedBenchmarkConfig(
        benchmark_offline_data_sources={"source": _source(camera_us=100_000)},  # pyright: ignore[reportArgumentType]
    )
    verify_policy_against_benchmark(_policy([45.0], past_camera_us=500_000), benchmark)
    with pytest.raises(ValueError, match="whole number of frames"):
        verify_policy_against_benchmark(_policy([45.0], past_camera_us=250_000), benchmark)


@dataclass
class _ClosedLoopBenchmarkConfig(AbstractBenchmarkConfig):
    @property
    @override
    def served_camera_interval_us(self) -> int:
        return 100_000

    @property
    @override
    def served_lidar_interval_us(self) -> int | None:
        return None


def test_closed_loop_benchmark_serves_declared_intervals() -> None:
    benchmark = _ClosedLoopBenchmarkConfig()
    verify_policy_against_benchmark(_policy([45.0], past_camera_us=500_000), benchmark)
    with pytest.raises(ValueError, match="whole number of frames"):
        verify_policy_against_benchmark(_policy([45.0], past_camera_us=250_000), benchmark)
    with pytest.raises(ValueError, match="serves no lidar"):
        verify_policy_against_benchmark(_policy([45.0], past_lidar_us=100_000), benchmark)


def test_benchmark_without_declared_intervals_skips_interval_checks() -> None:
    verify_policy_against_benchmark(_policy([45.0], past_lidar_us=100_000), _benchmark())


# -- verify_policy_against_offline_data_source --


def test_source_serves_whole_frame_intervals() -> None:
    verify_policy_against_offline_data_source(
        _policy([45.0], past_camera_us=500_000),
        _source(camera_us=100_000),
    )
    verify_policy_against_offline_data_source(
        _policy([45.0], past_lidar_us=100_000),
        _source(lidar_us=50_000),
    )
    verify_policy_against_offline_data_source(
        _policy([45.0], past_ego_state_us=200_000),
        _source(),
    )


def test_source_rejects_non_integer_frame_interval() -> None:
    with pytest.raises(ValueError, match="whole number of frames"):
        verify_policy_against_offline_data_source(
            _policy([45.0], past_camera_us=300_000),
            _source(camera_us=500_000),
        )
    with pytest.raises(ValueError, match="whole number of frames"):
        verify_policy_against_offline_data_source(
            _policy([45.0], past_lidar_us=50_000),
            _source(lidar_us=100_000),
        )
    with pytest.raises(ValueError, match="whole number of frames"):
        verify_policy_against_offline_data_source(
            _policy([45.0], past_ego_state_us=50_000),
            _source(),
        )


def test_source_rejects_missing_modality() -> None:
    with pytest.raises(ValueError, match="serves no lidar"):
        verify_policy_against_offline_data_source(
            _policy([45.0], past_lidar_us=100_000),
            _source(lidar_us=None),
        )


def test_source_ignores_anchor_only_policies() -> None:
    verify_policy_against_offline_data_source(_policy([45.0]), _source(lidar_us=None))


def test_source_serves_declared_target_point_range() -> None:
    bounded = AbstractOfflineDataSourceConfig(
        data_root="/data/123D",
        served_ego_state_interval_us=100_000,
        served_camera_interval_us=100_000,
        served_lidar_interval_us=None,
        min_served_target_point_distance_m=40.0,
        max_served_target_point_distance_m=80.0,
    )
    verify_policy_against_offline_data_source(_policy([45.0, 80.0]), bounded)
    with pytest.raises(ValueError, match="closer"):
        verify_policy_against_offline_data_source(_policy([20.0]), bounded)
    with pytest.raises(ValueError, match="farther"):
        verify_policy_against_offline_data_source(_policy([45.0, 90.0]), bounded)


# -- _verify_stream_records_declared_interval --


def _timestamps_us(period_us: int, count: int) -> list[int]:
    return [index * period_us for index in range(count)]


def test_leading_negative_timestamp_passes() -> None:
    _verify_stream_records_declared_interval(
        [t - 6_880 for t in _timestamps_us(100_000, 20)],
        interval_us=100_000,
        provider="the source at '/data/123D'",
        stream="camera FTCAM_F0",
    )


def test_matching_interval_passes() -> None:
    _verify_stream_records_declared_interval(
        _timestamps_us(100_000, 20),
        interval_us=100_000,
        provider="the source at '/data/123D'",
        stream="ego states",
    )


def test_wrong_interval_raises() -> None:
    with pytest.raises(ValueError, match=r"declares ego states every 50000 µs"):
        _verify_stream_records_declared_interval(
            _timestamps_us(100_000, 20),
            interval_us=50_000,
            provider="the source at '/data/123D'",
            stream="ego states",
        )


def test_dropped_frames_do_not_skew_the_period() -> None:
    timestamps = _timestamps_us(100_000, 30)
    with_holes = [t for index, t in enumerate(timestamps) if index % 4 != 3]
    _verify_stream_records_declared_interval(
        with_holes,
        interval_us=100_000,
        provider="the source at '/data/123D'",
        stream="camera PCAM_F0",
    )


def test_dropped_frames_warn(caplog: pytest.LogCaptureFixture) -> None:
    timestamps = _timestamps_us(100_000, 30)
    with_holes = [t for index, t in enumerate(timestamps) if index % 4 != 3]
    with caplog.at_level(logging.WARNING):
        _verify_stream_records_declared_interval(
            with_holes,
            interval_us=100_000,
            provider="the source at '/data/123D'",
            stream="camera PCAM_F0",
        )
    assert "drops 7 of 29" in caplog.text


def test_jitter_within_tolerance_passes() -> None:
    jitter_us = [0, 3000, -2000, 1000]
    timestamps = [t + jitter_us[index % 4] for index, t in enumerate(_timestamps_us(100_000, 20))]
    _verify_stream_records_declared_interval(
        timestamps,
        interval_us=100_000,
        provider="the source at '/data/123D'",
        stream="lidar LIDAR_TOP",
    )


def test_too_few_timestamps_pass_silently() -> None:
    _verify_stream_records_declared_interval(
        _timestamps_us(100_000, 3),
        interval_us=50_000,
        provider="the source at '/data/123D'",
        stream="ego states",
    )


# -- verify_offline_data_source_records_declared_intervals --


def _scene(ego_period_us: int) -> SceneAPI:
    scene = MagicMock(spec=SceneAPI)
    scene.get_all_ego_state_se3_timestamps.return_value = [
        Timestamp.from_us(t) for t in _timestamps_us(ego_period_us, 20)
    ]
    scene.available_camera_ids = []
    scene.available_lidar_ids = []
    return scene


def test_source_verifies_a_scene() -> None:
    source = AbstractOfflineDataSourceConfig(
        data_root="/data/123D",
        served_ego_state_interval_us=100_000,
        served_camera_interval_us=100_000,
        served_lidar_interval_us=None,
    )
    verify_offline_data_source_scenes_declared_intervals(source, _scene(ego_period_us=100_000))
    with pytest.raises(ValueError, match=r"the source at '/data/123D' declares ego states"):
        verify_offline_data_source_scenes_declared_intervals(source, _scene(ego_period_us=50_000))
