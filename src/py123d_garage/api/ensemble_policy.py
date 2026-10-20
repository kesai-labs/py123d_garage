# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import jaxtyping as jt
import torch
from py123d.api import SceneAPI
from py123d.geometry import Point2DIndex, PoseSE2Index
from torch import Tensor, nn
from typing_extensions import override

from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy, CacheTensorSpec
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    AbstractLabels,
    AbstractPredictions,
    NavigationConditioning,
)


@dataclass(frozen=True)
class EnsemblePredictions(AbstractPredictions):
    """The members' mean trajectory; the first member's bundle keeps the auxiliary heads."""

    # The members' mean (x, y, yaw) trajectory, with the yaw averaged on the circle.
    trajectory: jt.Float[Tensor, "*batch poses 3"]
    # The first member's predictions, which keep the auxiliary heads.
    first_member: AbstractPredictions

    @property
    @override
    def ego_trajectory_xy(self) -> jt.Float[Tensor, "*batch poses 2"]:
        """Inherited, see superclass."""
        return self.trajectory[..., : len(Point2DIndex)]

    @property
    @override
    def ego_trajectory_se2(self) -> jt.Float[Tensor, "*batch poses 3"]:
        """Inherited, see superclass."""
        return self.trajectory


class EnsemblePolicy(AbstractPolicy[AbstractFeatures, AbstractLabels, EnsemblePredictions]):
    """Averages the trajectories of several policies of one recipe, run one after the other."""

    def __init__(self, members: Sequence[AnyPolicy]) -> None:
        """
        Wraps initialized policies of one recipe; every call but forward goes to the first member.

        Args:
            members: the initialized policies, all with the same architecture and config.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.members = nn.ModuleList(members)

    @property
    def _first_member(self) -> AnyPolicy:
        """The member every call but forward delegates to."""
        return cast(AnyPolicy, self.members[0])

    @override
    def initialize(self, evaluation_checkpoint_file: str | None = None) -> None:
        """Does nothing, since the members are initialized before the ensemble wraps them."""
        assert evaluation_checkpoint_file is None, "the members load their checkpoints before the ensemble wraps them"

    @property
    @override
    def policy_config(self) -> AbstractPolicyConfig:
        """The first member's config."""
        return self._first_member.policy_config

    @override
    def build_model(self, from_local_checkpoint: bool) -> None:
        """Builds every member's model."""
        for member in self.members:
            cast(AnyPolicy, member).build_model(from_local_checkpoint)

    @override
    def forward(
        self,
        features: AbstractFeatures,
        navigation: NavigationConditioning,
    ) -> EnsemblePredictions:
        """
        Runs every member on the same features and averages the trajectories, the yaw circularly.

        Args:
            features: batched feature bundle from build_features.
            navigation: batched navigation conditioning.

        Returns:
            the mean trajectory, with the first member's bundle for the other heads.
        """
        member_predictions = [cast(AbstractPredictions, member(features, navigation)) for member in self.members]
        trajectories = torch.stack([predictions.ego_trajectory_se2 for predictions in member_predictions])
        yaw = trajectories[..., PoseSE2Index.YAW]
        mean_yaw = torch.atan2(torch.sin(yaw).mean(dim=0), torch.cos(yaw).mean(dim=0))
        trajectory = torch.cat(
            [trajectories[..., : len(Point2DIndex)].mean(dim=0), mean_yaw.unsqueeze(-1)],
            dim=-1,
        )
        return EnsemblePredictions(trajectory=trajectory, first_member=member_predictions[0])

    @override
    def build_features(self, scene_api: SceneAPI, cached: dict[str, Tensor]) -> AbstractFeatures:
        """Delegates to the first member."""
        return self._first_member.build_features(scene_api, cached)

    @override
    def build_labels(self, scene_api: SceneAPI, cached: dict[str, Tensor]) -> AbstractLabels:
        """Delegates to the first member."""
        return self._first_member.build_labels(scene_api, cached)

    @override
    def _permanent_cache_specs(self) -> dict[str, CacheTensorSpec]:
        """Delegates to the first member."""
        return self._first_member._permanent_cache_specs()  # pyright: ignore[reportPrivateUsage]

    @override
    def session_cache_names(self) -> frozenset[str]:
        """Delegates to the first member."""
        return self._first_member.session_cache_names()

    @override
    def compute_loss(self, labels: AbstractLabels, predictions: EnsemblePredictions) -> dict[str, Tensor]:
        """Delegates to the first member on its own predictions."""
        return self._first_member.compute_loss(labels, predictions.first_member)

    @override
    def loss_weights(self, epoch: int) -> dict[str, float]:
        """Delegates to the first member."""
        return self._first_member.loss_weights(epoch)

    @override
    def visualize_batch(
        self,
        features: AbstractFeatures,
        labels: AbstractLabels | None,
        navigation: NavigationConditioning,
        predictions: EnsemblePredictions,
        scene_api: SceneAPI,
    ) -> dict[str, Tensor]:
        """Delegates to the first member on its own predictions."""
        return self._first_member.visualize_batch(features, labels, navigation, predictions.first_member, scene_api)
