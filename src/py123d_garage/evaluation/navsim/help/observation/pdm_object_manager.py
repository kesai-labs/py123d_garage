# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import BoxDetectionSE2
from py123d.geometry import BoundingBoxSE2Index, Point2D
from py123d.geometry.utils.rotation_utils import normalize_angle

from py123d_garage.evaluation.navsim.help.utils.pdm_constants import (
    DYNAMIC_OBJECT_LABELS,
)

MAX_DYNAMIC_OBJECTS_PER_LABEL: dict[str, int] = {
    "vehicle": 50,
    "person": 25,
    "two_wheeler": 10,
    "else": 10,
}

MAX_STATIC_OBJECTS: int = 50


class PDMObjectManager:
    """Class that stores and sorts tracked objects around the ego-vehicle."""

    def __init__(
        self,
        max_dynamic_objects_per_label: dict[
            str,
            int,
        ] = MAX_DYNAMIC_OBJECTS_PER_LABEL,
        max_static_objects: int = MAX_STATIC_OBJECTS,
    ) -> None:
        """
        Constructor of PDMObjectManager.

        Args:
            max_dynamic_objects_per_label: cap on tracked dynamic objects kept per label
            max_static_objects: cap on tracked static objects kept
        """

        # all objects
        self._unique_objects: dict[str, BoxDetectionSE2] = {}

        # dynamic objects
        self._max_dynamic_objects_per_label = max_dynamic_objects_per_label
        self._dynamic_object_tokens: dict[str, list[str]] = {key: [] for key in max_dynamic_objects_per_label}
        self._dynamic_object_bbse2: dict[
            str,
            list[npt.NDArray[np.float64]],
        ] = {key: [] for key in max_dynamic_objects_per_label}
        self._dynamic_object_dxy: dict[str, list[npt.NDArray[np.float64]]] = {
            key: [] for key in max_dynamic_objects_per_label
        }

        # static objects
        self._max_static_objects = max_static_objects
        self._static_object_tokens: list[str] = []
        self._static_object_bbse2: list[npt.NDArray[np.float64]] = []

    @property
    def unique_objects(self) -> dict[str, BoxDetectionSE2]:
        """Mapping from track token to every object added to the manager."""
        return self._unique_objects

    def add_object(self, box_detection_se2: BoxDetectionSE2) -> None:
        """
        Adds an object to the manager, sorting it into the dynamic or static category.

        Args:
            box_detection_se2: any tracked object
        """

        bbse2_array = box_detection_se2.bounding_box_se2.array
        default_label = box_detection_se2.attributes.default_label
        track_token = box_detection_se2.attributes.track_token

        self._unique_objects[track_token] = box_detection_se2

        if default_label in DYNAMIC_OBJECT_LABELS:
            assert box_detection_se2.velocity_2d is not None, (
                f"Dynamic object {track_token} has no velocity information!"
            )
            velocity_2d = box_detection_se2.velocity_2d
            velocity_angle = np.arctan2(velocity_2d.y, velocity_2d.x)
            agent_drives_forward = (
                np.abs(
                    normalize_angle(
                        box_detection_se2.center_se2.yaw - velocity_angle,
                    ),
                )
                < np.pi / 2
            )

            track_heading = (
                box_detection_se2.center_se2.yaw
                if agent_drives_forward
                else normalize_angle(box_detection_se2.center_se2.yaw + np.pi)
            )

            dxy = np.array(
                [
                    np.cos(track_heading) * velocity_2d.magnitude,
                    np.sin(track_heading) * velocity_2d.magnitude,
                ],
                dtype=np.float64,
            ).T  # x,y velocity [m/s]

            label = default_label.serialize()
            label = label if label in self._dynamic_object_tokens else "else"

            self._dynamic_object_tokens[label].append(track_token)
            self._dynamic_object_bbse2[label].append(bbse2_array)
            self._dynamic_object_dxy[label].append(dxy)

        else:
            self._static_object_tokens.append(track_token)
            self._static_object_bbse2.append(bbse2_array)

    def get_nearest_objects(
        self,
        position: Point2D,
    ) -> tuple[
        list[str],
        jt.Float64[npt.NDArray[np.float64], "static 5"],
        list[str],
        jt.Float64[npt.NDArray[np.float64], "dynamic 5"],
        jt.Float64[npt.NDArray[np.float64], "dynamic 2"],
    ]:
        """
        Retrieves the nearest objects per category, capped per label.

        Args:
            position: global map position

        Returns:
            tuple containing tokens, bbse2, and dynamic information of objects
        """
        dynamic_object_tokens: list[str] = []
        dynamic_object_bbse2_list: list[npt.NDArray[np.float64]] = []
        dynamic_object_dxy_list: list[npt.NDArray[np.float64]] = []

        for dynamic_object_type in self._dynamic_object_tokens:
            (
                dynamic_object_tokens_,
                dynamic_object_bbse2_,
                dynamic_object_dxy_,
            ) = self._get_nearest_dynamic_objects(position, dynamic_object_type)

            if len(dynamic_object_tokens_) == 0:
                continue

            dynamic_object_tokens.extend(dynamic_object_tokens_)
            dynamic_object_bbse2_list.append(dynamic_object_bbse2_)
            dynamic_object_dxy_list.append(dynamic_object_dxy_)

        if len(dynamic_object_bbse2_list) > 0:
            dynamic_object_bbse2 = np.concatenate(
                dynamic_object_bbse2_list,
                axis=0,
                dtype=np.float64,
            )
            dynamic_object_dxy = np.concatenate(
                dynamic_object_dxy_list,
                axis=0,
                dtype=np.float64,
            )
        else:
            dynamic_object_bbse2 = np.empty(
                (0, len(BoundingBoxSE2Index)),
                dtype=np.float64,
            )
            dynamic_object_dxy = np.empty((0, 2), dtype=np.float64)

        static_object_tokens, static_object_bbse2_array = self._get_nearest_static_objects(position)

        return (
            static_object_tokens,
            static_object_bbse2_array,
            dynamic_object_tokens,
            dynamic_object_bbse2,
            dynamic_object_dxy,
        )

    def _get_nearest_dynamic_objects(
        self,
        position: Point2D,
        label: str,
    ) -> tuple[
        list[str],
        jt.Float64[npt.NDArray[np.float64], "objects 5"],
        jt.Float64[npt.NDArray[np.float64], "objects 2"],
    ]:
        """
        Retrieves the nearest dynamic objects of the given label.

        Args:
            position: Ego-vehicle position
            label: Object label to sort

        Returns:
            Tuple of tokens, bbse2, and velocity of nearest objects.
        """
        position_coords = position.array[None, ...]  # shape: (1,2)

        object_tokens = self._dynamic_object_tokens[label]
        if len(object_tokens) == 0:
            return (
                [],
                np.empty((0, len(BoundingBoxSE2Index)), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
            )

        object_bbse2 = np.array(
            self._dynamic_object_bbse2[label],
            dtype=np.float64,
        )
        object_dxy = np.array(self._dynamic_object_dxy[label], dtype=np.float64)

        position_to_center_dist = ((object_bbse2[..., BoundingBoxSE2Index.XY] - position_coords) ** 2.0).sum(
            axis=-1,
        ) ** 0.5

        object_argsort = np.argsort(position_to_center_dist)

        object_tokens = [object_tokens[int(i)] for i in object_argsort][: self._max_dynamic_objects_per_label[label]]
        object_bbse2 = object_bbse2[object_argsort][: self._max_dynamic_objects_per_label[label]]
        object_dxy = object_dxy[object_argsort][: self._max_dynamic_objects_per_label[label]]

        return (object_tokens, object_bbse2, object_dxy)

    def _get_nearest_static_objects(
        self,
        position: Point2D,
    ) -> tuple[list[str], jt.Float64[npt.NDArray[np.float64], "objects 5"]]:
        """
        Retrieves the nearest static obstacles around ego's position.

        Args:
            position: ego's position

        Returns:
            tuple of tokens and coords of nearest objects
        """
        position_coords = position.array[None, ...]  # shape: (1,2)

        object_tokens = self._static_object_tokens
        if len(object_tokens) == 0:
            return (
                [],
                np.empty((0, len(BoundingBoxSE2Index)), dtype=np.float64),
            )

        object_bbse2 = np.array(self._static_object_bbse2, dtype=np.float64)

        position_to_center_dist = ((object_bbse2[..., BoundingBoxSE2Index.XY] - position_coords) ** 2.0).sum(
            axis=-1,
        ) ** 0.5

        object_argsort = np.argsort(position_to_center_dist)

        object_tokens = [object_tokens[int(i)] for i in object_argsort][: self._max_static_objects]
        object_bbse2 = object_bbse2[object_argsort][: self._max_static_objects]

        return (object_tokens, object_bbse2)
