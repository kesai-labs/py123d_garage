from __future__ import annotations

import abc
import dataclasses

import jaxtyping as jt
import torch
from py123d.api import SceneAPI
from torch import Tensor

from py123d_garage.datatypes.numerics import PositiveFloat
from py123d_garage.datatypes.tensor import TensorBundle
from py123d_garage.py123d_help.scene_readers import get_target_points


@dataclasses.dataclass(frozen=True)
class NavigationConditioning(TensorBundle):
    """The navigation intent the policy is conditioned on."""

    target_points: jt.Float[Tensor, "*batch num_points 2"]

    @classmethod
    def from_scene(
        cls,
        scene_api: SceneAPI,
        target_point_distances_m: list[PositiveFloat],
    ) -> NavigationConditioning:
        """
        The conditioning of one scene; training and evaluation share this recipe.

        Args:
            scene_api: the scene to condition on.
            target_point_distances_m: arc-length distances along the log's route.

        Returns:
            the single-sample conditioning bundle.
        """
        return NavigationConditioning(
            target_points=torch.as_tensor(
                get_target_points(scene_api, target_point_distances_m),
            ),
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
