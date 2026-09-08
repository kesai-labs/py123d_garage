"""TensorBundle generic operations: apply, collate, flattening, and shape validation."""

from __future__ import annotations

from dataclasses import dataclass

import jaxtyping as jt
import pytest
import torch

from py123d_garage.api.abstract_policy_tensors import (
    AbstractFeatures,
    TensorBundle,
)
from py123d_garage.common.runtime_typing import typechecker


# Test modules are outside the package's import hook, so wrap explicitly.
@jt.jaxtyped(typechecker=typechecker)
@dataclass(frozen=True)
class _Boxes(TensorBundle):
    heatmap: jt.Float[torch.Tensor, "*batch 2 4 4"]


@jt.jaxtyped(typechecker=typechecker)
@dataclass(frozen=True)
class _Features(AbstractFeatures):
    camera: jt.Float[torch.Tensor, "*batch 3 8 8"]
    velocity: jt.Float[torch.Tensor, "*batch 1"]
    boxes: _Boxes | None = None
    lidar: jt.Float[torch.Tensor, "*batch 1 4 4"] | None = None


def _sample() -> _Features:
    return _Features(
        camera=torch.zeros(3, 8, 8),
        velocity=torch.zeros(1),
        boxes=_Boxes(heatmap=torch.zeros(2, 4, 4)),
    )


def test_construction_rejects_wrong_shape() -> None:
    with pytest.raises(Exception, match="camera"):
        _Features(camera=torch.zeros(4, 8, 8), velocity=torch.zeros(1))


def test_construction_rejects_inconsistent_batch_dims() -> None:
    with pytest.raises(Exception, match="velocity"):
        _Features(camera=torch.zeros(2, 3, 8, 8), velocity=torch.zeros(3, 1))


def test_named_tensors_flattens_nested_and_skips_none() -> None:
    names = [name for name, _ in _sample().named_tensors()]
    assert names == ["camera", "velocity", "heatmap"]


def test_as_dict_matches_named_tensors() -> None:
    sample = _sample()
    assert set(sample.as_dict()) == {"camera", "velocity", "heatmap"}
    assert sample.as_dict()["heatmap"].shape == (2, 4, 4)


def test_apply_maps_every_tensor_and_keeps_none() -> None:
    doubled = _sample().apply(lambda tensor: tensor + 1.0)
    assert isinstance(doubled, _Features)
    assert torch.all(doubled.camera == 1.0)
    assert doubled.boxes is not None
    assert torch.all(doubled.boxes.heatmap == 1.0)
    assert doubled.lidar is None


def test_collate_stacks_field_wise() -> None:
    batch = _Features.collate([_sample(), _sample(), _sample()])
    assert batch.camera.shape == (3, 3, 8, 8)
    assert batch.velocity.shape == (3, 1)
    assert batch.boxes is not None
    assert batch.boxes.heatmap.shape == (3, 2, 4, 4)
    assert batch.lidar is None
