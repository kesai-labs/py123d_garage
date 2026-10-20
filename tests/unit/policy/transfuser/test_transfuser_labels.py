# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""TransFuser label rasterization: CenterNet targets, box masks, and BEV coordinate transforms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
import torch
from py123d.api import SceneAPI
from py123d.datatypes import (
    CameraID,
    DefaultBoxDetectionLabel,
    EgoStateSE3,
    SegmentationCameraMetadata,
    Timestamp,
)
from py123d.datatypes.sensors.camera_segmentation_label import (
    CameraSegmentationLabel,
    DefaultCameraSegmentationLabel,
)
from py123d.geometry import BoundingBoxSE2Index, Point3D, PoseSE2
from shapely.geometry import Point
from typing_extensions import override

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.config.schema.policy.transfuser_config import (
    BevSemanticConfig,
    BoxDetectionConfig,
    CameraConfig,
    LidarConfig,
    PerspectiveConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser import features as features_module
from py123d_garage.policy.transfuser.labels import (
    _angle_to_class,
    _compute_box_mask,
    _compute_center_net_labels,
    _coords_to_pixel,
    _geometry_local_coords,
    _map_semantic,
    _relative_bounding_boxes,
    _within_z_gate,
    build_perspective_labels,
    merge_semantic_classes,
    semantic_classes,
)
from py123d_garage.policy.transfuser.network.decoder.center_net_decoder import (
    class2angle,
)

# One box row is (x, y, yaw, length, width); see BoundingBoxSE2Index.
_VEHICLE_BOX: npt.NDArray[np.float64] = np.array(
    [[10.5, -5.25, 0.3, 4.0, 2.0]],
    dtype=np.float64,
)


class _ExtendedLabel(CameraSegmentationLabel):
    ROAD = 1
    VEHICLE = 12
    ROAD_LINE = 15

    @override
    def to_default(self) -> DefaultCameraSegmentationLabel:
        """Folds ROAD_LINE into ROAD."""
        return DefaultCameraSegmentationLabel(1 if self is _ExtendedLabel.ROAD_LINE else self.value)


def test_semantic_mapping_rejects_unknown_ids() -> None:
    """A frame with ids outside the enum fails loudly instead of mapping silently."""
    road = DefaultCameraSegmentationLabel(1)
    image = np.array([[1, 14]], dtype=np.uint8)
    mapped = _map_semantic(image, DefaultCameraSegmentationLabel, None)
    assert mapped[0, 0] == int(road.to_default().value)
    with pytest.raises(ValueError, match=r"class ids \[200, 255\]"):
        _map_semantic(
            np.array([[1, 200], [14, 255]], dtype=np.uint8),
            DefaultCameraSegmentationLabel,
            None,
        )


def test_semantic_mapping_keeps_native_ids_of_the_configured_classes() -> None:
    """Without class ids the extra class folds; with them ids become positions and a foreign id is refused."""
    image = np.array([[15, 12]], dtype=np.uint8)
    assert _map_semantic(image, _ExtendedLabel, None).tolist() == [[1, 12]]
    assert _map_semantic(image, _ExtendedLabel, [1, 12, 15]).tolist() == [[2, 1]]
    with pytest.raises(ValueError, match=r"semantic_class_ids \[15\] are no DefaultCameraSegmentationLabel members"):
        _map_semantic(image, DefaultCameraSegmentationLabel, [1, 12, 15])
    with pytest.raises(ValueError, match=r"class ids \[15\]"):
        _map_semantic(image, _ExtendedLabel, [1, 12])


def test_angle_to_class_hand_computed_bins() -> None:
    """12 bins of pi/6: bin centers get residual 0, in-between angles keep the exact remainder."""
    assert _angle_to_class(0.0, 12) == (0, 0.0)
    bin_idx, residual = _angle_to_class(np.pi / 6.0, 12)
    assert bin_idx == 1
    assert abs(residual) < 1e-12
    bin_idx, residual = _angle_to_class(0.2, 12)
    assert bin_idx == 0
    assert np.isclose(residual, 0.2)
    bin_idx, residual = _angle_to_class(-0.2, 12)
    assert bin_idx == 0
    assert np.isclose(residual, -0.2)
    bin_idx, residual = _angle_to_class(np.pi, 12)
    assert bin_idx == 6
    assert abs(residual) < 1e-12


def test_angle_to_class_round_trips_through_class2angle() -> None:
    angles = [-3.0, -1.0, -0.2, 0.0, 0.7, 2.0, 3.1]
    encoded = [_angle_to_class(angle, 12) for angle in angles]
    decoded = class2angle(
        torch.tensor([[bin_idx for bin_idx, _ in encoded]]),
        torch.tensor([[residual for _, residual in encoded]]),
        num_dir_bins=12,
    )
    assert torch.allclose(
        decoded,
        torch.tensor([angles], dtype=torch.float32),
        atol=1e-6,
    )


def test_coords_to_pixel_maps_extents_to_raster() -> None:
    """Column indexes x from bev_min_x, row indexes y from bev_min_y, at 4 px/m."""
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
    )
    coords = np.array(
        [[0.0, 0.0], [10.25, -5.5], [-32.0, -40.0], [0.3, 0.0]],
        dtype=np.float64,
    )
    pixel = _coords_to_pixel(coords, config)
    assert pixel.dtype == np.int32
    assert pixel.tolist() == [[128, 160], [169, 138], [0, 0], [129, 160]]


def test_center_net_labels_single_vehicle() -> None:
    """A vehicle at (10.5, -5.25) on the 4x-downsampled 96x80 grid lands at col 42, row 34."""
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        lidar_config=LidarConfig(),
    )
    targets = _compute_center_net_labels(
        _VEHICLE_BOX,
        [DefaultBoxDetectionLabel.VEHICLE],
        np.array([3.0]),
        config,
    )
    heatmap = targets.center_net_heatmap
    assert heatmap.shape == (4, 80, 96)
    assert heatmap[0, 34, 42] == 1.0
    assert int(torch.argmax(heatmap[0])) == 34 * 96 + 42
    assert torch.all(heatmap[1:] == 0)
    # 1 px/m after downsampling: sizes in meters, offsets are the sub-pixel remainders.
    assert targets.center_net_wh[:, 34, 42].tolist() == [4.0, 2.0]
    assert targets.center_net_offset[:, 34, 42].tolist() == [0.5, 0.75]
    assert float(targets.center_net_wh.sum()) == 6.0
    assert int(targets.center_net_yaw_class[34, 42]) == 1
    assert torch.isclose(
        targets.center_net_yaw_res[0, 34, 42],
        torch.tensor(0.3 - np.pi / 6.0, dtype=torch.float32),
    )
    assert float(targets.center_net_pixel_weight[:, 34, 42].sum()) == 2.0
    assert float(targets.center_net_pixel_weight.sum()) == 2.0
    assert float(targets.center_net_avg_factor) == 1.0
    assert targets.center_net_velocity is None


def test_center_net_labels_route_classes_to_channels() -> None:
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
    )
    boxes = np.concatenate(
        [_VEHICLE_BOX, [[0.0, 0.0, 0.0, 0.8, 0.8]]],
        axis=0,
    )
    targets = _compute_center_net_labels(
        boxes,
        [DefaultBoxDetectionLabel.VEHICLE, DefaultBoxDetectionLabel.PERSON],
        np.array([3.0, 1.0]),
        config,
    )
    heatmap = targets.center_net_heatmap
    assert heatmap[0, 34, 42] == 1.0
    assert heatmap[1, 40, 32] == 1.0
    assert heatmap[1, 34, 42] == 0.0
    assert heatmap[0, 40, 32] == 0.0
    assert float(targets.center_net_avg_factor) == 2.0


def test_center_net_labels_skip_out_of_bounds_and_unmapped_boxes() -> None:
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
    )
    boxes = np.array(
        [
            [100.0, 0.0, 0.0, 4.0, 2.0],  # beyond bev_max_x
            [-40.0, 0.0, 0.0, 4.0, 2.0],  # behind bev_min_x
            [0.0, 0.0, 0.0, 1.0, 1.0],  # ANIMAL is in no detection class
        ],
        dtype=np.float64,
    )
    targets = _compute_center_net_labels(
        boxes,
        [
            DefaultBoxDetectionLabel.VEHICLE,
            DefaultBoxDetectionLabel.VEHICLE,
            DefaultBoxDetectionLabel.ANIMAL,
        ],
        np.zeros(3),
        config,
    )
    assert float(targets.center_net_avg_factor) == 0.0
    assert torch.all(targets.center_net_heatmap == 0)
    assert torch.all(targets.center_net_pixel_weight == 0)


def test_center_net_velocity_only_with_sweep_history() -> None:
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        box_detection_config=BoxDetectionConfig(predict_box_velocity=True),
        lidar_config=LidarConfig(
            lidar_interval_us=100_000,
            lidar_horizon_us=100_000,
        ),
    )
    targets = _compute_center_net_labels(
        _VEHICLE_BOX,
        [DefaultBoxDetectionLabel.VEHICLE],
        np.array([3.0]),
        config,
    )
    velocity = targets.center_net_velocity
    assert velocity is not None
    assert velocity[0, 34, 42] == 3.0
    assert float(velocity.sum()) == 3.0


def test_box_mask_covers_the_footprint() -> None:
    """A 4x2 m axis-aligned vehicle at the origin fills exactly its footprint pixels."""
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
    )
    mask = _compute_box_mask(
        np.array([[0.0, 0.0, 0.0, 4.0, 2.0]], dtype=np.float64),
        [DefaultBoxDetectionLabel.VEHICLE],
        [DefaultBoxDetectionLabel.VEHICLE],
        config,
    )
    assert mask.shape == (320, 384)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    # x in [-2, 2] -> cols 120..136, y in [-1, 1] -> rows 156..164 at 4 px/m.
    assert (rows.min(), rows.max()) == (156, 164)
    assert (cols.min(), cols.max()) == (120, 136)
    assert mask[160, 128]


def test_box_mask_filters_by_label() -> None:
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
    )
    mask = _compute_box_mask(
        np.array([[0.0, 0.0, 0.0, 4.0, 2.0]], dtype=np.float64),
        [DefaultBoxDetectionLabel.PERSON],
        [DefaultBoxDetectionLabel.VEHICLE],
        config,
    )
    assert not mask.any()


def test_box_mask_scales_pedestrians() -> None:
    """A 0.6x0.6 m person scaled 10x rasterizes as the 6x6 m footprint."""
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        bev_semantic_config=BevSemanticConfig(pedestrian_bev_extent_scale=10.0),
    )
    mask = _compute_box_mask(
        np.array([[0.0, 0.0, 0.0, 0.6, 0.6]], dtype=np.float64),
        [DefaultBoxDetectionLabel.PERSON],
        [DefaultBoxDetectionLabel.PERSON],
        config,
    )
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert (rows.min(), rows.max()) == (148, 172)
    assert (cols.min(), cols.max()) == (116, 140)


def test_geometry_local_coords_is_world_to_ego() -> None:
    """A point ahead of the pose maps to +x, a point to its right maps to -y."""
    origin = PoseSE2(2.0, 3.0, np.pi / 2.0)
    ahead = cast(Point, _geometry_local_coords(Point(2.0, 4.0), origin))
    assert np.allclose([ahead.x, ahead.y], [1.0, 0.0], atol=1e-9)
    right = cast(Point, _geometry_local_coords(Point(3.0, 3.0), origin))
    assert np.allclose([right.x, right.y], [0.0, -1.0], atol=1e-9)


def test_semantic_class_merges_fold_and_renumber() -> None:
    """Merged classes fold into their targets and the rest renumber densely; unknown ids are refused."""
    road, sidewalk, terrain, vehicle = (
        int(DefaultCameraSegmentationLabel[name].value) for name in ("ROAD", "SIDEWALK", "TERRAIN", "VEHICLE")
    )
    config = TransfuserConfig(
        perspective_config=PerspectiveConfig(
            use_semantic=True,
            semantic_class_merges={sidewalk: road, terrain: 0},
        ),
    )
    classes = semantic_classes(config)
    assert len(classes) == len(DefaultCameraSegmentationLabel) - 2
    assert sidewalk not in classes
    label = torch.tensor([[[sidewalk, terrain, vehicle]]], dtype=torch.uint8)
    merged = merge_semantic_classes(label, config)
    assert merged.dtype == torch.uint8
    assert merged.tolist() == [[[classes.index(road), 0, classes.index(vehicle)]]]
    assert merge_semantic_classes(label, TransfuserConfig()) is label
    with pytest.raises(AssertionError, match=r"\[99\]"):
        semantic_classes(
            TransfuserConfig(perspective_config=PerspectiveConfig(semantic_class_merges={99: road})),
        )


def test_semantic_class_merges_on_native_ids() -> None:
    """With native class ids the label stores positions, so merges renumber those positions."""
    config = TransfuserConfig(
        perspective_config=PerspectiveConfig(
            use_semantic=True,
            semantic_class_ids=[0, 1, 8, 15],
            semantic_class_merges={8: 0},
        ),
    )
    assert semantic_classes(config) == [0, 1, 15]
    label = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.uint8)
    assert merge_semantic_classes(label, config).tolist() == [[[0, 1, 0, 2]]]


def test_within_z_gate_uses_the_vertex_nearest_in_xy() -> None:
    """A ramp descending to the ego is kept; an overpass above the ego is dropped."""
    ego = Point3D(0.0, 0.0, 0.0)
    ramp = np.array([[0.0, 0.0, 0.5], [30.0, 0.0, 6.0]], dtype=np.float64)
    overpass = np.array([[-30.0, 0.0, 6.0], [0.0, 0.0, 6.0], [30.0, 0.0, 6.0]], dtype=np.float64)
    assert _within_z_gate(ramp, ego, 3.0)
    assert not _within_z_gate(overpass, ego, 3.0)
    assert _within_z_gate(overpass, Point3D(0.0, 0.0, 6.0), 3.0)


def test_frame_without_box_annotations_has_no_boxes() -> None:
    ego = MagicMock(spec=EgoStateSE3)
    ego.rear_axle_se2 = PoseSE2(0.0, 0.0, 0.0)
    boxes, labels, speeds = _relative_bounding_boxes(None, ego)
    assert boxes.shape == (0, len(BoundingBoxSE2Index))
    assert labels == []
    assert speeds.shape == (0,)


class _PerspectiveScene:
    """SceneAPI stand-in whose semantic frames hold one class per recording time."""

    def __init__(self, class_by_time_us: dict[int, int], anchor_us: int) -> None:
        self._class_by_time_us = class_by_time_us
        self._anchor_us = anchor_us

    @property
    def scene_metadata(self) -> SimpleNamespace:
        return SimpleNamespace(dataset="nuplan")

    @property
    def scene_uuid(self) -> str:
        return "test-scene"

    @property
    def log_name(self) -> str:
        return "test-log"

    def frame(self, time_us: int) -> SimpleNamespace:
        return SimpleNamespace(
            image=np.zeros((8, 16, 3), dtype=np.uint8),
            metadata=None,
            timestamp=Timestamp.from_us(time_us),
        )

    def get_camera_at_iteration(self, iteration: int, camera_id: CameraID) -> SimpleNamespace:
        return self.frame(self._anchor_us)

    def _semantic(self, time_us: int) -> SimpleNamespace | None:
        if time_us not in self._class_by_time_us:
            return None
        metadata = MagicMock(spec=SegmentationCameraMetadata)
        metadata.segmentation_label_class = DefaultCameraSegmentationLabel
        image = np.full((8, 16), self._class_by_time_us[time_us], dtype=np.uint8)
        return SimpleNamespace(image=image, metadata=metadata, timestamp=Timestamp.from_us(time_us))

    def get_camera_semantic_at_iteration(self, iteration: int, camera_id: CameraID) -> SimpleNamespace | None:
        return self._semantic(self._anchor_us)

    def get_camera_semantic_at_timestamp(
        self,
        timestamp: Timestamp,
        camera_id: CameraID,
        criteria: str = "exact",
    ) -> SimpleNamespace | None:
        assert criteria == "exact"
        return self._semantic(timestamp.time_us)


SceneAPI.register(_PerspectiveScene)


def test_perspective_labels_follow_the_camera_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each label row comes from the frame its camera row shows; a row before the rollout began repeats the oldest."""
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(500_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        camera_config=CameraConfig(
            input_cameras={"nuplan": [CameraID.PCAM_F0]},
            image_width=64,
            image_height=96,
            past_camera_offset_us=500_000,
            num_past_camera_rows=2,
        ),
        perspective_config=PerspectiveConfig(use_semantic=True, use_depth=False),
    )
    anchor_us = 1_000_000
    old, mid, now = (int(member.value) for member in list(DefaultCameraSegmentationLabel)[1:4])
    scene = _PerspectiveScene({anchor_us - 1_000_000: old, anchor_us - 500_000: mid, anchor_us: now}, anchor_us)
    num_label_rows = 96 // config.perspective_config.perspective_downsample_factor

    def row_classes(past: list[SimpleNamespace | None]) -> list[int]:
        monkeypatch.setattr(features_module, "sample_past_camera", lambda *_, **__: past)
        semantic, depth = build_perspective_labels(cast(SceneAPI, scene), config)
        assert depth is None
        assert semantic is not None
        assert semantic.shape == (num_label_rows, 64 // config.perspective_config.perspective_downsample_factor)
        rows = semantic.reshape(3, num_label_rows // 3, -1)
        assert all(bool((row == row[0, 0]).all()) for row in rows)
        return [int(row[0, 0]) for row in rows]

    assert row_classes([scene.frame(anchor_us - 500_000), scene.frame(anchor_us - 1_000_000)]) == [old, mid, now]
    assert row_classes([scene.frame(anchor_us - 500_000), None]) == [mid, mid, now]
