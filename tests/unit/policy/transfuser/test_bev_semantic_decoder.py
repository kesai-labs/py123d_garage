"""BEV semantic loss and metrics: a map-less sample's non-box pixels may be background or any map class."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from py123d_garage.config.schema.policy.transfuser_config import (
    BevSemanticConfig,
    TransfuserConfig,
)
from py123d_garage.policy.transfuser.network.bev_semantic_decoder import (
    BEVSemanticDecoder,
    BevSemanticLabels,
)

# Menu ids 1..3 are map classes, 4..6 boxes; the decoder re-indexes them densely.
_NON_BOX_CLASSES = [0, 1, 2, 3]
_BOX_CLASSES = [4, 5, 6]
_ROAD, _VEHICLE = 1, 5


def _decoder(class_ids: list[int]) -> BEVSemanticDecoder:
    config = TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
        bev_semantic_config=BevSemanticConfig(bev_semantic_class_ids=class_ids),
    )
    return BEVSemanticDecoder(config)


def _labels(target: torch.Tensor, has_map: list[bool]) -> BevSemanticLabels:
    return BevSemanticLabels(bev_semantic=target, bev_semantic_has_map=torch.tensor(has_map))


def _random_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits over 7 classes and targets drawn from background and the box classes, as a map-less raster has."""
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(batch_size, 7, 4, 5, generator=generator, requires_grad=True)
    target = torch.tensor([0, *_BOX_CLASSES])[torch.randint(0, 4, (batch_size, 4, 5), generator=generator)]
    return logits, target


def test_map_less_loss_sums_probability_over_non_box_classes_on_background() -> None:
    decoder = _decoder([1, 2, 3, 4, 5, 6])
    logits, target = _random_batch(2)
    log_probs = F.log_softmax(logits, dim=1)
    non_box = torch.logsumexp(log_probs[:, _NON_BOX_CLASSES], dim=1)
    per_pixel = torch.where(target == 0, -non_box, -log_probs.gather(1, target[:, None]).squeeze(1))

    loss = decoder.compute_loss(logits, _labels(target, [False, False]))["loss_bev_semantic"]

    assert loss.item() == pytest.approx(per_pixel.mean().item())


def test_map_less_background_loss_only_depends_on_the_non_box_mass() -> None:
    """Uniform logits: -log(4/7) without a map, -log(1/7) with one."""
    decoder = _decoder([1, 2, 3, 4, 5, 6])
    logits = torch.zeros((1, 7, 1, 1))
    target = torch.zeros((1, 1, 1), dtype=torch.int64)

    without_map = decoder.compute_loss(logits, _labels(target, [False]))["loss_bev_semantic"]
    with_map = decoder.compute_loss(logits, _labels(target, [True]))["loss_bev_semantic"]

    assert without_map.item() == pytest.approx(-math.log(4 / 7))
    assert with_map.item() == pytest.approx(-math.log(1 / 7))


def test_map_less_sample_is_free_to_call_background_road() -> None:
    decoder = _decoder([1, 2, 3, 4, 5, 6])
    logits = torch.full((1, 7, 1, 1), -20.0)
    logits[0, _ROAD] = 20.0
    target = torch.zeros((1, 1, 1), dtype=torch.int64)

    without_map = decoder.compute_loss(logits, _labels(target, [False]))["loss_bev_semantic"]
    with_map = decoder.compute_loss(logits, _labels(target, [True]))["loss_bev_semantic"]

    assert without_map.item() == pytest.approx(0.0, abs=1e-3)
    assert with_map.item() > 10.0


def test_map_less_sample_still_penalizes_boxes_on_background() -> None:
    decoder = _decoder([1, 2, 3, 4, 5, 6])
    logits = torch.full((1, 7, 1, 1), -20.0)
    logits[0, _VEHICLE] = 20.0
    target = torch.zeros((1, 1, 1), dtype=torch.int64)

    loss = decoder.compute_loss(logits, _labels(target, [False]))["loss_bev_semantic"]

    assert loss.item() > 10.0


def test_samples_with_a_map_use_plain_cross_entropy_in_a_mixed_batch() -> None:
    decoder = _decoder([1, 2, 3, 4, 5, 6])
    logits, target = _random_batch(2)
    target[1, 0, 0] = _ROAD

    mixed = decoder.compute_loss(logits, _labels(target, [False, True]))["loss_bev_semantic"]
    map_less_half = decoder.compute_loss(logits[:1], _labels(target[:1], [False]))["loss_bev_semantic"]
    mapped_half = F.cross_entropy(logits[1:], target[1:])

    assert mixed.item() == pytest.approx((map_less_half.item() + mapped_half.item()) / 2)


def test_box_only_classes_are_unaffected_by_the_flag() -> None:
    decoder = _decoder([4, 5, 6])
    logits = torch.zeros((1, 4, 1, 1))
    target = torch.zeros((1, 1, 1), dtype=torch.int64)

    without_map = decoder.compute_loss(logits, _labels(target, [False]))["loss_bev_semantic"]
    with_map = decoder.compute_loss(logits, _labels(target, [True]))["loss_bev_semantic"]

    assert without_map.item() == pytest.approx(-math.log(1 / 4))
    assert with_map.item() == pytest.approx(-math.log(1 / 4))


def test_metrics_count_map_predictions_as_background_without_a_map() -> None:
    decoder = _decoder([1, 2, 3, 4, 5, 6])
    logits = torch.full((1, 7, 1, 1), -20.0)
    logits[0, _ROAD] = 20.0
    target = torch.zeros((1, 1, 1), dtype=torch.int64)

    without_map = decoder.compute_metrics(logits, _labels(target, [False]))
    with_map = decoder.compute_metrics(logits, _labels(target, [True]))

    assert without_map["bev_semantic_miou"].item() == pytest.approx(1.0)
    assert with_map["bev_semantic_miou"].item() == pytest.approx(0.0)
