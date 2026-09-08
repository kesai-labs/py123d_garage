# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from py123d.datatypes import DefaultBoxDetectionLabel

DYNAMIC_OBJECT_LABELS = {
    DefaultBoxDetectionLabel.VEHICLE,
    DefaultBoxDetectionLabel.PERSON,
    DefaultBoxDetectionLabel.TWO_WHEELER,
    DefaultBoxDetectionLabel.ANIMAL,
    DefaultBoxDetectionLabel.TRAIN,
    DefaultBoxDetectionLabel.OTHER,
}
