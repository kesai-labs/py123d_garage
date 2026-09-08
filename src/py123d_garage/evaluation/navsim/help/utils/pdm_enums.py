# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

# TODO: Move & rename this file for common usage (not specific for PDM)
from __future__ import annotations

from enum import IntEnum

from py123d.common.utils.enums import classproperty


class StateIndex(IntEnum):
    """Index mapping for array representation of ego states."""

    X = 0
    Y = 1
    HEADING = 2
    VELOCITY_X = 3
    VELOCITY_Y = 4
    ACCELERATION_X = 5
    ACCELERATION_Y = 6
    STEERING_ANGLE = 7
    STEERING_RATE = 8
    ANGULAR_VELOCITY = 9
    ANGULAR_ACCELERATION = 10

    @classproperty
    def POINT(cls) -> slice:
        # assumes X, Y have subsequent indices
        return slice(cls.X, cls.Y + 1)

    @classproperty
    def STATE_SE2(cls) -> slice:
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls.X, cls.HEADING + 1)

    @classproperty
    def VELOCITY_2D(cls) -> slice:
        # assumes velocity X, Y have subsequent indices
        return slice(cls.VELOCITY_X, cls.VELOCITY_Y + 1)

    @classproperty
    def ACCELERATION_2D(cls) -> slice:
        # assumes acceleration X, Y have subsequent indices
        return slice(cls.ACCELERATION_X, cls.ACCELERATION_Y + 1)


class PointIndex(IntEnum):
    """Index mapping for (x,y) arrays."""

    X = 0
    Y = 1


class SE2Index(IntEnum):
    """Index mapping for state se2 (x,y,θ) arrays."""

    X = 0
    Y = 1
    HEADING = 2


class DynamicStateIndex(IntEnum):
    """Index mapping for dynamic car state (output of controller)."""

    ACCELERATION_X = 0
    STEERING_RATE = 1


class StateIDMIndex(IntEnum):
    """Index mapping for IDM states."""

    PROGRESS = 0
    VELOCITY = 1


class LeadingAgentIndex(IntEnum):
    """Index mapping for leading agent state (for IDM policies)."""

    PROGRESS = 0
    VELOCITY = 1
    LENGTH_REAR = 2


class BBCoordsIndex(IntEnum):
    """Index mapping for corners and center of bounding boxes."""

    FRONT_LEFT = 0
    REAR_LEFT = 1
    REAR_RIGHT = 2
    FRONT_RIGHT = 3
    CENTER = 4


class EgoAreaIndex(IntEnum):
    """Index mapping for area of ego agent (used in PDMScorer)."""

    MULTIPLE_LANES = 0
    NON_DRIVABLE_AREA = 1
    ONCOMING_TRAFFIC = 2
    INTERSECTION = 3


class MultiMetricIndex(IntEnum):
    """Index mapping multiplicative metrics (used in PDMScorer)."""

    NO_COLLISION = 0
    DRIVABLE_AREA = 1
    TRAFFIC_LIGHT_COMPLIANCE = 2
    DRIVING_DIRECTION = 3


class WeightedMetricIndex(IntEnum):
    """Index mapping weighted metrics (used in PDMScorer)."""

    PROGRESS = 0
    TTC = 1
    COMFORT = 2
