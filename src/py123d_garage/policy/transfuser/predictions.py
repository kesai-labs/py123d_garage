# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""TransFuser's typed prediction bundle."""

from __future__ import annotations

from dataclasses import dataclass

import jaxtyping as jt
import torch
from py123d.geometry import Point2DIndex, PoseSE2Index
from typing_extensions import override

from py123d_garage.api.abstract_policy_tensors import AbstractPredictions
from py123d_garage.policy.transfuser.network.decoder.center_net_decoder import (
    CenterNetBoundingBoxPrediction,
)


@dataclass(frozen=True)
class TransfuserPredictions(AbstractPredictions):
    """TransFuser's output tensors; fields of disabled heads are None."""

    # Planning decoder output per pose, (x, y, yaw) first; while autograd is on, the path points follow the poses.
    trajectory: jt.Float[torch.Tensor, "*batch poses columns"] | None = None
    # (x, y, route heading) of the route points ahead; only set when autograd is off.
    path: jt.Float[torch.Tensor, "*batch path_points 3"] | None = None
    # The action decoder's waypoint columns; set only outside autograd and when it predicts waypoints.
    waypoints: jt.Float[torch.Tensor, "*batch poses 3"] | None = None
    # The action decoder's unicycle rollout columns; set only outside autograd.
    rollout: jt.Float[torch.Tensor, "*batch poses 3"] | None = None
    # Perspective semantic logits.
    semantic: (
        jt.Float[
            torch.Tensor,
            "*batch semantic_classes semantic_height semantic_width",
        ]
        | None
    ) = None
    # Perspective depth in meters.
    depth: jt.Float[torch.Tensor, "*batch semantic_height semantic_width"] | None = None
    # BEV semantic logits.
    bev_semantic: jt.Float[torch.Tensor, "*batch bev_classes bev_height bev_width"] | None = None
    # CenterNet box head outputs.
    boxes: CenterNetBoundingBoxPrediction | None = None

    @property
    @override
    def ego_trajectory_xy(self) -> jt.Float[torch.Tensor, "*batch poses 2"]:
        """Inherited, see superclass."""
        assert self.trajectory is not None, "The planning decoder is disabled, so there is no ego trajectory!"
        return self.trajectory[..., : len(Point2DIndex)]

    @property
    @override
    def ego_trajectory_se2(self) -> jt.Float[torch.Tensor, "*batch poses 3"]:
        """Inherited, see superclass."""
        assert self.trajectory is not None, "The planning decoder is disabled, so there is no ego trajectory!"
        return self.trajectory[..., : len(PoseSE2Index)]

    @property
    @override
    def plan_outputs(self) -> dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        outputs = {"waypoints": self.waypoints, "rollout": self.rollout, "path": self.path}
        return {name: output for name, output in outputs.items() if output is not None}
