"""TransFuser label rasterization: CenterNet targets, box masks, and BEV coordinate transforms."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from py123d.datatypes import DefaultBoxDetectionLabel
from py123d.datatypes.sensors.camera_segmentation_label import (
    DefaultCameraSegmentationLabel,
)
from py123d.geometry import PoseSE2
from shapely.geometry import Point

from py123d_garage.config.schema.policy.transfuser_config import (
    BevSemanticConfig,
    BoxDetectionConfig,
    LidarConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.labels import (
    _angle_to_class,
    _compute_box_mask,
    _compute_center_net_labels,
    _coords_to_pixel,
    _geometry_local_coords,
    _map_semantic_to_default,
)
from py123d_garage.policy.transfuser.network.center_net_decoder import (
    class2angle,
)

# One box row is (x, y, yaw, length, width); see BoundingBoxSE2Index.
_VEHICLE_BOX: npt.NDArray[np.float64] = np.array(
    [[10.5, -5.25, 0.3, 4.0, 2.0]],
    dtype=np.float64,
)


def test_semantic_mapping_rejects_unknown_ids() -> None:
    """A frame with ids outside the enum fails loudly instead of mapping silently."""
    road = DefaultCameraSegmentationLabel(1)
    image = np.array([[1, 14]], dtype=np.uint8)
    mapped = _map_semantic_to_default(image, DefaultCameraSegmentationLabel)
    assert mapped[0, 0] == int(road.to_default().value)
    with pytest.raises(ValueError, match=r"class ids \[200, 255\]"):
        _map_semantic_to_default(
            np.array([[1, 200], [14, 255]], dtype=np.uint8),
            DefaultCameraSegmentationLabel,
        )


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
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
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
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
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
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
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
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
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
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
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
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
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
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
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
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
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
