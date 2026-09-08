"""The label classes of lead's CARLA 123D export, so its logs open without lead installed."""

from __future__ import annotations

from py123d.datatypes import (
    DefaultBoxDetectionLabel,
    DefaultCameraSegmentationLabel,
)
from py123d.datatypes.detections.box_detection_label import (
    BOX_DETECTION_LABEL_REGISTRY,  # pyright: ignore[reportUnknownVariableType]
    BoxDetectionLabel,
    register_box_detection_label,  # pyright: ignore[reportUnknownVariableType]
)
from py123d.datatypes.sensors.camera_segmentation_label import (
    CAMERA_SEGMENTATION_LABEL_REGISTRY,  # pyright: ignore[reportUnknownVariableType]
    CameraSegmentationLabel,
    register_camera_segmentation_label,  # pyright: ignore[reportUnknownVariableType]
)
from typing_extensions import override


@register_box_detection_label
class CarlaBoxDetectionLabel(BoxDetectionLabel):
    """Native LEAD/CARLA box classes, copied 1:1 from lead's exporter."""

    CAR = 0
    WALKER = 1
    BICYCLE = 2
    TRAFFIC_LIGHT = 3
    STOP_SIGN = 4
    STATIC = 5
    STATIC_PROP_CAR = 6
    TRAFFIC_LIGHT_PHYSICAL = 7
    STOP_SIGN_PHYSICAL = 8
    TRAFFIC_SIGN = 9

    @override
    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        # TODO@ln2697: This mapping is a best-effort guess, and may be wrong in some cases.
        return {
            CarlaBoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            CarlaBoxDetectionLabel.WALKER: DefaultBoxDetectionLabel.PERSON,
            CarlaBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            CarlaBoxDetectionLabel.TRAFFIC_LIGHT: DefaultBoxDetectionLabel.TRAFFIC_LIGHT,
            CarlaBoxDetectionLabel.STOP_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            CarlaBoxDetectionLabel.STATIC: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            CarlaBoxDetectionLabel.STATIC_PROP_CAR: DefaultBoxDetectionLabel.VEHICLE,
            CarlaBoxDetectionLabel.TRAFFIC_LIGHT_PHYSICAL: DefaultBoxDetectionLabel.TRAFFIC_LIGHT,
            CarlaBoxDetectionLabel.STOP_SIGN_PHYSICAL: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            CarlaBoxDetectionLabel.TRAFFIC_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
        }[self]


@register_camera_segmentation_label
class CarlaCameraSegmentationLabel(CameraSegmentationLabel):
    """Per-pixel semantic classes of LEAD 123D logs, copied 1:1 from lead's exporter."""

    UNLABELED = 0
    ROADS = 1
    SIDEWALKS = 2
    BUILDING = 3
    WALL = 4
    FENCE = 5
    POLE = 6
    TRAFFIC_LIGHT = 7
    TRAFFIC_SIGN = 8
    VEGETATION = 9
    TERRAIN = 10
    SKY = 11
    PEDESTRIAN = 12
    RIDER = 13
    CAR = 14
    TRUCK = 15
    BUS = 16
    TRAIN = 17
    MOTORCYCLE = 18
    BICYCLE = 19
    STATIC = 20
    DYNAMIC = 21
    OTHER = 22
    WATER = 23
    ROAD_LINE = 24
    GROUND = 25
    BRIDGE = 26
    RAIL_TRACK = 27
    GUARD_RAIL = 28

    @override
    def to_default(self) -> DefaultCameraSegmentationLabel:
        """Inherited, see superclass."""
        # TODO@ln2697: This mapping is a best-effort guess, and may be wrong in some cases.
        return {
            CarlaCameraSegmentationLabel.UNLABELED: DefaultCameraSegmentationLabel.IGNORE,
            CarlaCameraSegmentationLabel.ROADS: DefaultCameraSegmentationLabel.ROAD,
            CarlaCameraSegmentationLabel.SIDEWALKS: DefaultCameraSegmentationLabel.SIDEWALK,
            CarlaCameraSegmentationLabel.BUILDING: DefaultCameraSegmentationLabel.BUILDING,
            CarlaCameraSegmentationLabel.WALL: DefaultCameraSegmentationLabel.BUILDING,
            CarlaCameraSegmentationLabel.FENCE: DefaultCameraSegmentationLabel.BUILDING,
            CarlaCameraSegmentationLabel.POLE: DefaultCameraSegmentationLabel.POLE,
            CarlaCameraSegmentationLabel.TRAFFIC_LIGHT: DefaultCameraSegmentationLabel.TRAFFIC_LIGHT,
            CarlaCameraSegmentationLabel.TRAFFIC_SIGN: DefaultCameraSegmentationLabel.TRAFFIC_SIGN,
            CarlaCameraSegmentationLabel.VEGETATION: DefaultCameraSegmentationLabel.VEGETATION,
            CarlaCameraSegmentationLabel.TERRAIN: DefaultCameraSegmentationLabel.TERRAIN,
            CarlaCameraSegmentationLabel.SKY: DefaultCameraSegmentationLabel.SKY,
            CarlaCameraSegmentationLabel.PEDESTRIAN: DefaultCameraSegmentationLabel.PERSON,
            CarlaCameraSegmentationLabel.RIDER: DefaultCameraSegmentationLabel.RIDER,
            CarlaCameraSegmentationLabel.CAR: DefaultCameraSegmentationLabel.VEHICLE,
            CarlaCameraSegmentationLabel.TRUCK: DefaultCameraSegmentationLabel.VEHICLE,
            CarlaCameraSegmentationLabel.BUS: DefaultCameraSegmentationLabel.VEHICLE,
            CarlaCameraSegmentationLabel.TRAIN: DefaultCameraSegmentationLabel.VEHICLE,
            CarlaCameraSegmentationLabel.MOTORCYCLE: DefaultCameraSegmentationLabel.TWO_WHEELER,
            CarlaCameraSegmentationLabel.BICYCLE: DefaultCameraSegmentationLabel.TWO_WHEELER,
            CarlaCameraSegmentationLabel.STATIC: DefaultCameraSegmentationLabel.OTHER,
            CarlaCameraSegmentationLabel.DYNAMIC: DefaultCameraSegmentationLabel.OTHER,
            CarlaCameraSegmentationLabel.OTHER: DefaultCameraSegmentationLabel.OTHER,
            CarlaCameraSegmentationLabel.WATER: DefaultCameraSegmentationLabel.TERRAIN,
            CarlaCameraSegmentationLabel.ROAD_LINE: DefaultCameraSegmentationLabel.ROAD,
            CarlaCameraSegmentationLabel.GROUND: DefaultCameraSegmentationLabel.TERRAIN,
            CarlaCameraSegmentationLabel.BRIDGE: DefaultCameraSegmentationLabel.BUILDING,
            CarlaCameraSegmentationLabel.RAIL_TRACK: DefaultCameraSegmentationLabel.OTHER,
            CarlaCameraSegmentationLabel.GUARD_RAIL: DefaultCameraSegmentationLabel.BUILDING,
        }[self]


# HACK: Read LEAD's data without requiring lead to be installed.
BOX_DETECTION_LABEL_REGISTRY["lead.api.py123d_log_api.CarlaBoxDetectionLabel"] = CarlaBoxDetectionLabel
CAMERA_SEGMENTATION_LABEL_REGISTRY["lead.api.py123d_log_api.CarlaCameraSegmentationLabel"] = (
    CarlaCameraSegmentationLabel
)
