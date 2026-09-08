"""
The data-access assumptions both policies rest on, checked against every dataset on disk.

The unit tests for py123d_garage.py123d_help.scene_readers run against SceneAPI doubles, so they encode
what we believe py123d does rather than check it. These read real logs of every dataset
under PY123D_GARAGE_DATA_ROOT: whether the search criteria behave as assumed, and whether
the half-interval tolerance both policies pass survives the jitter actually recorded.

The sensor rigs differ per dataset (pinhole PCAM against f-theta FTCAM), which is exactly
where the assumptions are most likely to part company with reality.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from py123d.api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution import ThreadPoolExecutor
from py123d.datatypes import CameraID, LidarID
from scipy.stats import spearmanr

from py123d_garage.py123d_help.misc.constants import FRONT_CAMERAS
from py123d_garage.py123d_help.scene_builders.scene_builder import build_scene_builder
from py123d_garage.py123d_help.scene_readers import (
    InsufficientRouteError,
    get_target_points,
    interpolate_ego_se2_at,
    lidar_at_anchor,
    lidar_at_timestamp,
    sample_past_camera,
    sample_past_lidar,
)

pytestmark = pytest.mark.slow

LOG = logging.getLogger(__name__)

_NUM_SCENES = 50
_HISTORY_US = 1_500_000
# How far the ego moved from the anchor against how much the sample changed, ranked within
# each stack and then taken across stacks. Measured medians: nuplan 0.99 lidar / 1.00 camera,
# kesai 1.00 / 1.00, physical-ai-av 0.93 / 1.00, the last one lower because it covers 53 m
# per stack, far enough that the change saturates.
_MIN_TRAVEL_CORRELATION = 0.5
_MIN_CORRELATION_SCENES = 10
# No stream is sampled finer than the grid the policies themselves ask for.
_MIN_SAMPLING_INTERVAL_US = 100_000
# The bird's-eye window a sweep is summarized over.
_SWEEP_EXTENT_M = 32.0
# Reported speed against a centred difference of the poses; slack for the differencing.
_MAX_SPEED_DISAGREEMENT = 0.15


def _dataset_roots() -> list[str]:
    """The 123D dirs under PY123D_GARAGE_DATA_ROOT, one per dataset that is on disk."""
    root = os.environ.get("PY123D_GARAGE_DATA_ROOT")
    if not root:
        return []
    return sorted(str(path.parent) for path in Path(root).glob("*/123D/logs") if path.is_dir())


@pytest.fixture(scope="module", params=_dataset_roots() or [None], ids=lambda p: Path(p).parent.name if p else "none")
def scenes(request: pytest.FixtureRequest) -> Sequence[SceneAPI]:
    data_root = request.param
    if data_root is None:
        pytest.skip("no 123D datasets on disk (set PY123D_GARAGE_DATA_ROOT to the data dir)")
    # py123d discovers a split only if its directory name says train, val or test, which
    # skips whole datasets that name their splits otherwise; list them off disk instead.
    split_names = sorted(path.name for path in (Path(data_root) / "logs").iterdir() if path.is_dir())
    scene_filter = SceneFilter(history_duration_s=2.0, future_duration_s=5.0, split_names=split_names)
    scene_filter.max_num_scenes = _NUM_SCENES
    built = build_scene_builder(data_root, lazy=False).get_scenes(
        filter=scene_filter,
        executor=ThreadPoolExecutor(max_workers=4),
    )
    if not built:
        pytest.skip(f"{data_root} carries no scene passing the filter")
    return built


def _interval_jitter_and_dropouts(timestamps_us: list[int]) -> tuple[int, int, int]:
    """
    The stream's median interval, its jitter, and how many samples went missing.

    A dropped sample leaves a gap of roughly two intervals, which would otherwise read as
    a full interval of jitter. Gaps beyond 1.5 intervals are counted as dropouts instead,
    so the jitter reflects the recording clock rather than the missing frames.
    """
    gaps = np.diff(np.asarray(sorted(timestamps_us), dtype=np.int64))
    interval_us = int(np.median(gaps))
    is_single = gaps < 1.5 * interval_us
    jitter_us = int(np.abs(gaps[is_single] - interval_us).max()) if is_single.any() else 0
    dropouts = int(np.sum(np.round(gaps[~is_single] / interval_us) - 1)) if (~is_single).any() else 0
    return interval_us, jitter_us, dropouts


def _front_camera_id(scene: SceneAPI) -> CameraID:
    """The camera filling the front slot of whichever dataset the scene came from."""
    return FRONT_CAMERAS[scene.scene_metadata.dataset][1]


def _native_interval_us(timestamps_us: list[int]) -> int:
    """
    The interval a stream actually records at, floored at the policies' own 100 ms grid.

    Sampling a stream more coarsely than it records throws away the resolution the
    correlation is measured over: a 4 Hz camera asked for a 2 Hz grid yields three points
    across a 1.5 s window, and a rank correlation over three points can only take the
    values +-1 and +-0.5.
    """
    gaps = np.diff(np.asarray(sorted(timestamps_us), dtype=np.int64))
    return max(int(np.median(gaps)), _MIN_SAMPLING_INTERVAL_US)


def _lidar_timestamps_us(scene: SceneAPI) -> list[int]:
    """Every recorded LiDAR timestamp, from whichever sensor the dataset carries."""
    merged = scene.get_all_lidar_timestamps(LidarID.LIDAR_MERGED, True)
    top = scene.get_all_lidar_timestamps(LidarID.LIDAR_TOP, True)
    return [timestamp.time_us for timestamp in (merged or top)]


def test_recorded_jitter_fits_the_half_interval_tolerance(scenes: Sequence[SceneAPI]) -> None:
    """
    Both policies serve a target with anything inside half an interval.

    That constant is only sound if a stream stays within half a period of its own nominal
    grid, which is what this measures per dataset rather than assumes. Missing samples are
    counted and logged rather than asserted on: a dataset is allowed to drop sensors, and
    both policies already have a branch for it. A simulated dataset records on a fixed
    timestep and jitters by nothing at all, which is a pass, not a stream to skip.
    """
    dataset = scenes[0].scene_metadata.dataset
    worst: dict[str, tuple[int, int]] = {}
    dropouts: dict[str, int] = {}
    for scene in scenes:
        streams = {
            "camera": [t.time_us for t in scene.get_all_camera_timestamps(_front_camera_id(scene), True)],
            "lidar": _lidar_timestamps_us(scene),
            "ego": [t.time_us for t in scene.get_all_ego_state_se3_timestamps(True)],
        }
        for name, timestamps_us in streams.items():
            if len(timestamps_us) < 3:
                continue
            interval_us, jitter_us, dropped = _interval_jitter_and_dropouts(timestamps_us)
            dropouts[name] = dropouts.get(name, 0) + dropped
            if name not in worst or jitter_us > worst[name][1]:
                worst[name] = (interval_us, jitter_us)

    assert worst, f"{dataset} carried no stream with enough samples to measure"
    for name, (interval_us, jitter_us) in sorted(worst.items()):
        LOG.info(
            f"{dataset} {name}: interval {interval_us} µs, worst jitter {jitter_us} µs, "
            f"{dropouts.get(name, 0)} samples missing across {len(scenes)} scenes",
        )
        assert jitter_us <= interval_us // 2, (
            f"{dataset} {name} jitters {jitter_us} µs against a {interval_us} µs interval, so "
            f"the half-interval tolerance the policies pass is too tight for this dataset."
        )


def test_backward_criteria_separates_a_dropout_from_a_young_stream(scenes: Sequence[SceneAPI]) -> None:
    """
    The whole drop-versus-too-early branch rests on this one behaviour.

    The past-stream sampler raises on a miss only when a backward search finds something, so
    'backward' must find nothing before a stream's first sample and something after it.
    """
    checked = 0
    for scene in scenes:
        recorded_us = [t.time_us for t in scene.get_all_lidar_timestamps(LidarID.LIDAR_MERGED, True)]
        if len(recorded_us) < 2:
            continue
        assert lidar_at_timestamp(scene, min(recorded_us) - 10_000_000, "backward") is None, (
            "a backward search before the first recorded sweep must find nothing, otherwise "
            "a rollout younger than the context is mistaken for a sensor dropout"
        )
        assert lidar_at_timestamp(scene, max(recorded_us), "backward") is not None
        checked += 1

    if not checked:
        pytest.skip(f"{scenes[0].scene_metadata.dataset} carries no LiDAR to check the criteria with")


def test_target_points_sit_at_the_requested_arc_lengths(scenes: Sequence[SceneAPI]) -> None:
    """
    The target points are interpolated along the route, so their spacing along it should
    match what was asked for. Straight-line distance is a lower bound on arc length, so a
    point may sit nearer than requested on a bend but never further.
    """
    dataset = scenes[0].scene_metadata.dataset
    distances_m = [10.0, 20.0, 40.0]
    measured, missing_route = 0, 0
    for scene in scenes:
        try:
            target_points = get_target_points(scene, distances_m)
        except InsufficientRouteError:
            missing_route += 1
            continue
        assert target_points.shape == (len(distances_m), 2)
        straight_line_m = np.linalg.norm(target_points.astype(np.float64), axis=1)
        assert np.all(np.diff(straight_line_m) > 0.0), (
            f"{dataset} target points are not ordered by distance: {straight_line_m}"
        )
        assert np.all(straight_line_m <= np.array(distances_m) + 1e-3), (
            f"{dataset} target points sit further than their arc length: {straight_line_m} against {distances_m}"
        )
        measured += 1

    LOG.info(f"{dataset} target points: {measured} scenes measured, {missing_route} without a usable route")
    if not measured:
        pytest.skip(f"{dataset} carries no scene with a route long enough for {distances_m[-1]} m")


def _image_fingerprint(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """
    A 32x32 exposure-invariant summary of one frame.

    Block-averaging drops sensor noise, and centring then scaling drops the brightness and
    gain changes that otherwise dominate a raw pixel difference.
    """
    gray = image.astype(np.float64).mean(axis=2)
    block_y, block_x = gray.shape[0] // 32, gray.shape[1] // 32
    small = gray[: block_y * 32, : block_x * 32].reshape(32, block_y, 32, block_x).mean(axis=(1, 3))
    centred = small - small.mean()
    norm = float(np.linalg.norm(centred))
    return centred / norm if norm else centred


def _sweep_fingerprint(xyz: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """
    A 32x32 bird's-eye occupancy summary of one sweep.

    A sweep's bounding box hardly moves as the ego drives, because the sensor sees roughly
    the same radius wherever it is; what changes is where the points sit inside it. Binning
    the points into an ego-frame grid measures that, so the summary tracks the drive the way
    an image does.
    """
    xy = xyz[:, :2].astype(np.float64)
    inside = np.all(np.abs(xy) < _SWEEP_EXTENT_M, axis=1)
    grid, _, _ = np.histogram2d(
        xy[inside, 0],
        xy[inside, 1],
        bins=32,
        range=[[-_SWEEP_EXTENT_M, _SWEEP_EXTENT_M], [-_SWEEP_EXTENT_M, _SWEEP_EXTENT_M]],
    )
    centred = grid - grid.mean()
    norm = float(np.linalg.norm(centred))
    return centred / norm if norm else centred


def _served_context(
    samples: list[object | None],
    anchor: object,
    dataset: str,
) -> list[object]:
    """The served part of a context stack, oldest first, with its ordering asserted."""
    served = [sample for sample in samples if sample is not None] + [anchor]
    timestamps_us = [sample.timestamp.time_us for sample in served]  # pyright: ignore[reportAttributeAccessIssue]
    assert timestamps_us == sorted(timestamps_us), f"{dataset} served its context out of order: {timestamps_us}"
    return served


def test_interpolated_ego_poses_move_at_a_plausible_speed(scenes: Sequence[SceneAPI]) -> None:
    """
    Interpolated ego poses must trace a drive, not teleport.

    This is the one stream with an absolute bound rather than a relative one: the poses are
    global, so consecutive ones an interval apart are separated by speed times interval. A
    frame mix-up or a botched interpolation shows up as a jump no car could make, which no
    relative comparison would catch.
    """
    dataset = scenes[0].scene_metadata.dataset
    interval_us = 100_000
    max_speed_mps = 60.0
    fastest_mps = 0.0
    measured = 0
    for scene in scenes:
        anchor_us = scene.get_ego_state_se3_at_iteration(0)
        if anchor_us is None:
            continue
        targets_us = anchor_us.timestamp.time_us + np.arange(-10, 1, dtype=np.int64) * interval_us
        try:
            poses_se2 = interpolate_ego_se2_at(scene, targets_us, boundary_tolerance_us=interval_us)
        except ValueError:
            continue

        steps_m = np.linalg.norm(np.diff(poses_se2[:, :2], axis=0), axis=1)
        speeds_mps = steps_m / (interval_us / 1e6)
        fastest_mps = max(fastest_mps, float(speeds_mps.max()))
        assert speeds_mps.max() <= max_speed_mps, (
            f"{dataset} ego poses move at {speeds_mps.max():.1f} m/s between samples "
            f"{interval_us} µs apart; no vehicle does that, so the poses are not one drive."
        )
        measured += 1

    if not measured:
        pytest.skip(f"{dataset} carries no ego history long enough to interpolate")
    LOG.info(f"{dataset} ego: fastest interpolated step {fastest_mps:.1f} m/s over {measured} scenes")


def _drive_tracking_correlation(
    fingerprints: list[npt.NDArray[np.float64]],
    poses_se2: npt.NDArray[np.float64],
) -> float:
    """
    How well a stack's change away from its anchor tracks how far the ego moved away from it.

    Args:
        fingerprints: one summary per sample, oldest first, the anchor's last
        poses_se2: the ego pose at each sample's own recorded timestamp, same order

    Returns:
        the rank correlation, or nan if every sample sits the same distance from the anchor
    """
    distances = [float(np.abs(fingerprint - fingerprints[-1]).mean()) for fingerprint in fingerprints[:-1]]
    if len(set(distances)) == 1:
        return float("nan")
    travelled_m = [float(np.linalg.norm(pose_se2[:2] - poses_se2[-1, :2])) for pose_se2 in poses_se2[:-1]]
    return float(spearmanr(travelled_m, distances).statistic)


def test_camera_frames_change_with_how_far_the_ego_drove(scenes: Sequence[SceneAPI]) -> None:
    """
    A stack must change away from its anchor in step with how far the ego drove.

    Each frame is compared against the anchor of its own stack, so the scene stays fixed
    and only the ego's displacement varies. Correlating one number per scene across scenes
    does not work here: the fixture draws its scenes from one or two logs at a near-constant
    speed, so the travel barely varies and the correlation ranks noise. The correlation is
    ranked rather than linear because the change saturates once two frames decorrelate.

    This holds the fetchers to what the SceneAPI doubles cannot check: that the frame a
    timestamp search returns really was recorded at the timestamp it reports. It does not
    tell one camera from another, and it is blind to the order the stack came back in,
    because every frame is paired with the ego pose at its own timestamp; the order is
    asserted off the timestamps instead, and identical neighbours are rejected outright
    because a search collapsed onto one frame would otherwise correlate perfectly.
    """
    dataset = scenes[0].scene_metadata.dataset
    camera_id = _front_camera_id(scenes[0])
    interval_us = _native_interval_us(
        [timestamp.time_us for timestamp in scenes[0].get_all_camera_timestamps(camera_id, True)],
    )
    correlations: list[float] = []
    collapsed = 0
    for scene in scenes:
        camera_id = _front_camera_id(scene)
        anchor = scene.get_camera_at_iteration(0, camera_id)
        if anchor is None:
            continue
        try:
            frames = reversed(
                sample_past_camera(
                    scene_api=scene,
                    camera_id=camera_id,
                    anchor_timestamp_us=anchor.timestamp.time_us,
                    horizon_us=_HISTORY_US,
                    interval_us=interval_us,
                ),
            )
            served = _served_context(list(frames), anchor, dataset)
            if len(served) < 3:
                continue
            timestamps_us = np.array(
                [frame.timestamp.time_us for frame in served],  # pyright: ignore[reportAttributeAccessIssue]
                dtype=np.int64,
            )
            poses_se2 = interpolate_ego_se2_at(scene, timestamps_us, boundary_tolerance_us=interval_us)
        except ValueError:
            continue

        fingerprints = [_image_fingerprint(frame.image) for frame in served]  # pyright: ignore[reportAttributeAccessIssue]
        for older, newer in pairwise(fingerprints):
            assert not np.array_equal(older, newer), (
                f"{dataset} served the same frame twice in one stack; the timestamp search is "
                f"collapsing onto a single frame."
            )
        correlation = _drive_tracking_correlation(fingerprints, poses_se2)
        if math.isnan(correlation):
            collapsed += 1
            continue
        correlations.append(correlation)

    if collapsed and not correlations:
        pytest.fail(f"{dataset} frames never change within a stack across {collapsed} scenes; the stacks are collapsed")
    if len(correlations) < _MIN_CORRELATION_SCENES:
        pytest.skip(f"{dataset} yielded {len(correlations)} comparable stacks, too few to correlate")

    median = float(np.median(correlations))
    LOG.info(
        f"{dataset} camera: travel-to-change rank correlation, median {median:+.2f} over {len(correlations)} stacks",
    )
    assert median > _MIN_TRAVEL_CORRELATION, (
        f"{dataset} frame change does not track how far the ego drove (median correlation "
        f"{median:+.2f}); the frames are not following the drive."
    )


def test_reported_ego_speed_agrees_with_the_distance_travelled(scenes: Sequence[SceneAPI]) -> None:
    """
    The speed a log reports and the distance between its poses must be the same drive.

    Both come out of the log independently: the speed off the dynamic state, the distance
    off the interpolated poses. TransFuser feeds the first to the policy and the second is
    what every trajectory is measured against, so a disagreement means one of them is in
    the wrong frame, the wrong unit, or off by a timestamp.
    """
    dataset = scenes[0].scene_metadata.dataset
    interval_us = 100_000
    interval_s = interval_us / 1e6
    reported_mps: list[float] = []
    measured_mps: list[float] = []
    for scene in scenes:
        anchor = scene.get_ego_state_se3_at_iteration(0)
        if anchor is None or anchor.dynamic_state_se3 is None:
            continue
        targets_us = anchor.timestamp.time_us + np.array([-interval_us, interval_us], dtype=np.int64)
        try:
            poses_se2 = interpolate_ego_se2_at(scene, targets_us, boundary_tolerance_us=interval_us)
        except ValueError:
            continue

        # Centred difference around the anchor, so it lines up with the reported speed there.
        measured_mps.append(float(np.linalg.norm(poses_se2[1, :2] - poses_se2[0, :2]) / (2.0 * interval_s)))
        reported_mps.append(float(np.linalg.norm(anchor.dynamic_state_se3.velocity_2d.array)))

    if len(reported_mps) < _MIN_CORRELATION_SCENES:
        pytest.skip(f"{dataset} yielded {len(reported_mps)} ego samples, too few to compare")

    reported = np.array(reported_mps)
    measured = np.array(measured_mps)
    moving = reported > 1.0
    if not moving.any():
        pytest.skip(f"{dataset} carries no moving scene to compare speeds on")

    relative_error = float(np.median(np.abs(measured[moving] - reported[moving]) / reported[moving]))
    LOG.info(
        f"{dataset} ego: reported against measured speed, median relative error "
        f"{relative_error:.1%} over {int(moving.sum())} moving scenes",
    )
    assert relative_error < _MAX_SPEED_DISAGREEMENT, (
        f"{dataset} reports a speed that disagrees with its own poses by {relative_error:.1%}; "
        f"one of the two is in the wrong frame or off by a timestamp."
    )


def test_lidar_sweeps_change_with_how_far_the_ego_drove(scenes: Sequence[SceneAPI]) -> None:
    """The camera test's counterpart for LiDAR, on bird's-eye occupancy instead of pixels."""
    dataset = scenes[0].scene_metadata.dataset
    interval_us = _native_interval_us(_lidar_timestamps_us(scenes[0]))
    correlations: list[float] = []
    collapsed = 0
    for scene in scenes:
        anchor = lidar_at_anchor(scene)
        if anchor is None:
            continue
        try:
            sweeps = reversed(
                sample_past_lidar(scene, anchor.timestamp.time_us, _HISTORY_US, interval_us),
            )
            served = _served_context(list(sweeps), anchor, dataset)
            if len(served) < 3:
                continue
            timestamps_us = np.array(
                [sweep.timestamp.time_us for sweep in served],  # pyright: ignore[reportAttributeAccessIssue]
                dtype=np.int64,
            )
            poses_se2 = interpolate_ego_se2_at(scene, timestamps_us, boundary_tolerance_us=interval_us)
        except ValueError:
            continue

        fingerprints = [_sweep_fingerprint(sweep.xyz) for sweep in served]  # pyright: ignore[reportAttributeAccessIssue]
        for older, newer in pairwise(fingerprints):
            assert not np.array_equal(older, newer), (
                f"{dataset} served the same sweep twice in one stack; the timestamp search is "
                f"collapsing onto a single sweep."
            )
        correlation = _drive_tracking_correlation(fingerprints, poses_se2)
        if math.isnan(correlation):
            collapsed += 1
            continue
        correlations.append(correlation)

    if collapsed and not correlations:
        pytest.fail(f"{dataset} sweeps never change within a stack across {collapsed} scenes; the stacks are collapsed")
    if len(correlations) < _MIN_CORRELATION_SCENES:
        pytest.skip(f"{dataset} yielded {len(correlations)} comparable sweep stacks, too few to correlate")

    median = float(np.median(correlations))
    LOG.info(
        f"{dataset} lidar: travel-to-change rank correlation, median {median:+.2f} over {len(correlations)} stacks",
    )
    assert median > _MIN_TRAVEL_CORRELATION, (
        f"{dataset} sweep change does not track how far the ego drove (median correlation "
        f"{median:+.2f}); the sweeps are not following the drive."
    )
