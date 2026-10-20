# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from py123d.api import SceneAPI

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import BackboneConfig, TransfuserConfig
from py123d_garage.policy.transfuser import transfuser_policy
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy


def _policy() -> TransfuserPolicy:
    config = TransfuserConfig(
        trajectory_offsets_us=uniform_offsets_us(100_000, 4_000_000),
        required_target_point_distances_m=[45.0],
        backbone_config=BackboneConfig(image_encoder_pretrained=False),
    )
    return TransfuserPolicy(PolicyConfig(transfuser_config=config))


def _session_tensors(policy: TransfuserPolicy) -> dict[str, torch.Tensor]:
    """What the node-local session cache serves without a permanent store, at the policy's shapes."""
    config = policy.policy_config
    path_config = config.planning_config.path_config
    assert path_config is not None
    shapes = {
        "trajectory": (config.trajectory_num_steps, 3),
        "action": (config.trajectory_num_steps, 2),
        "path": (path_config.num_points, 3),
        "velocity": (1,),
        "velocity_vector": (2,),
        "acceleration_vector": (2,),
        "target_points": (len(config.required_target_point_distances_m), 2),
        "target_point_headings": (len(config.required_target_point_distances_m),),
    }
    return {name: torch.zeros(shapes[name]) for name in policy.session_cache_names()}


def _store_labels() -> dict[str, torch.Tensor]:
    """The BEV label tensors a permanent store serves, at a 2 x 2 raster."""
    return {
        "bev_semantic": torch.zeros(2, 2, dtype=torch.uint8),
        "bev_semantic_has_map": torch.tensor(True),
        "center_net_heatmap": torch.zeros(3, 2, 2),
        "center_net_wh": torch.zeros(2, 2, 2),
        "center_net_offset": torch.zeros(2, 2, 2),
        "center_net_yaw_class": torch.zeros(2, 2, dtype=torch.uint8),
        "center_net_yaw_res": torch.zeros(1, 2, 2),
        "center_net_pixel_weight": torch.zeros(2, 2, 2),
        "center_net_avg_factor": torch.tensor(1.0),
    }


def test_session_tensors_alone_build_the_bev_labels_live(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy()
    scenes_built: list[SceneAPI] = []

    def build_bev_labels(scene_api: SceneAPI, config: TransfuserConfig) -> tuple[None, None]:
        del config
        scenes_built.append(scene_api)
        return None, None

    monkeypatch.setattr(transfuser_policy, "build_bev_labels", build_bev_labels)
    scene = cast(SceneAPI, MagicMock(spec=SceneAPI))
    cached = _session_tensors(policy)

    labels = policy.build_labels(scene, cached)

    assert scenes_built == [scene]
    assert labels.trajectory is cached["trajectory"]
    assert labels.bev_semantic is None
    assert labels.center_net is None


def test_stored_bev_labels_are_read_not_rebuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy()
    monkeypatch.setattr(transfuser_policy, "build_bev_labels", MagicMock(side_effect=AssertionError("rebuilt")))
    cached = _session_tensors(policy) | _store_labels()

    labels = policy.build_labels(cast(SceneAPI, MagicMock(spec=SceneAPI)), cached)

    assert labels.bev_semantic is not None
    assert labels.center_net is not None
    assert torch.equal(labels.bev_semantic.bev_semantic, cached["bev_semantic"])
    assert torch.equal(labels.center_net.center_net_heatmap, cached["center_net_heatmap"])
