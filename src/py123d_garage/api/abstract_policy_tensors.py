# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import abc
import dataclasses

import jaxtyping as jt
import torch
from py123d.api import SceneAPI
from torch import Tensor

from py123d_garage.datatypes.numerics import NonNegativeFloat
from py123d_garage.datatypes.tensor import TensorBundle
from py123d_garage.py123d_help.scene_readers import get_target_point_poses


@dataclasses.dataclass(frozen=True)
class NavigationConditioning(TensorBundle):
    """The navigation intent the policy is conditioned on."""

    # Target point (x, y) positions in the anchor rear-axle frame, one per target point distance; NaN for no intent.
    target_points: jt.Float[Tensor, "*batch num_points 2"]
    # Route direction at each target point, radians in the same rear-axle frame.
    target_point_headings: jt.Float[Tensor, "*batch num_points"]

    @classmethod
    def from_scene(
        cls,
        scene_api: SceneAPI,
        target_point_distances_m: list[NonNegativeFloat],
        nan_past_route_end: bool = False,
    ) -> NavigationConditioning:
        """
        Builds the navigation conditioning of one scene for training and evaluation alike.

        Args:
            scene_api: the scene to condition on.
            target_point_distances_m: arc-length distances along the log's route.
            nan_past_route_end: whether points past the route's end become NaN instead of raising.

        Returns:
            the single-sample conditioning bundle.
        """
        target_point_poses = torch.as_tensor(
            get_target_point_poses(scene_api, target_point_distances_m, nan_past_route_end),
        )
        return NavigationConditioning(
            target_points=target_point_poses[:, :2],
            target_point_headings=target_point_poses[:, 2],
        )


@dataclasses.dataclass(frozen=True)
class AbstractFeatures(TensorBundle):
    """The model's input tensors; each policy declares its concrete fields."""


@dataclasses.dataclass(frozen=True)
class AbstractLabels(TensorBundle):
    """The supervision tensors; each policy declares its concrete fields."""


@dataclasses.dataclass(frozen=True)
class AbstractPredictions(TensorBundle, abc.ABC):
    """The model's output tensors; each policy declares its concrete fields."""

    @property
    @abc.abstractmethod
    def ego_trajectory_xy(self) -> jt.Float[Tensor, "*batch poses 2"]:
        """The predicted future ego trajectory as relative xy positions."""

    @property
    @abc.abstractmethod
    def ego_trajectory_se2(self) -> jt.Float[Tensor, "*batch poses 3"]:
        """The predicted future ego trajectory as relative (x, y, yaw) poses."""

    @property
    def plan_outputs(self) -> dict[str, Tensor]:
        """The planning outputs besides the driven trajectory, by name, for logging; none unless a policy declares some."""
        return {}
