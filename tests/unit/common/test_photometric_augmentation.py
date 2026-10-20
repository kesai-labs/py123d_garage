# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Unit tests for the photometric camera augmentation, run on CPU tensors."""

from __future__ import annotations

import dataclasses
from dataclasses import fields

import pytest
import torch

from py123d_garage.common.sensor.photometric_augmentation import augment_camera_batch
from py123d_garage.config.schema.training.photometric_augmentation_config import PhotometricAugmentationConfig

_EFFECTS = [
    config_field.name
    for config_field in fields(PhotometricAugmentationConfig)
    if config_field.name.endswith("_probability")
]


def _uint8_batch(height: int, width: int, seed: int = 0) -> torch.Tensor:
    """
    Draws a reproducible mid-range uint8 camera batch.

    Args:
        height: image height.
        width: image width.
        seed: generator seed for the pixel draw.

    Returns:
        a (4, 3, height, width) uint8 batch.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(32, 224, (4, 3, height, width), generator=generator, dtype=torch.uint8)


def _only(probabilities: dict[str, float], max_effects_per_category: int = 1) -> PhotometricAugmentationConfig:
    """
    A config with every effect off except the given ones.

    Args:
        probabilities: chance per effect field name.
        max_effects_per_category: the cap on effects per category.

    Returns:
        the config.
    """
    off = dict.fromkeys(_EFFECTS, 0.0)
    return PhotometricAugmentationConfig(
        **{**off, **probabilities},
        max_effects_per_category=max_effects_per_category,
        max_effects_per_camera=99,
        clean_fraction=0.0,
    )


def _augment(
    rgb: torch.Tensor,
    config: PhotometricAugmentationConfig,
    num_tiles: int,
    num_rows: int,
) -> torch.Tensor:
    """
    Runs the augmentation at the config's effect probabilities.

    Args:
        rgb: the camera batch.
        config: the effect probabilities and the cap.
        num_tiles: cameras side by side in a row.
        num_rows: frames stacked top to bottom.

    Returns:
        the augmented batch.
    """
    return augment_camera_batch(
        rgb,
        config.effect_probabilities,
        config.max_effects_per_category,
        config.max_effects_per_camera,
        config.clean_fraction,
        num_tiles=num_tiles,
        num_rows=num_rows,
    )


def test_all_effects_off_is_exact_identity() -> None:
    rgb = _uint8_batch(32, 96)
    out = _augment(rgb, _only({}), num_tiles=3, num_rows=1)
    assert torch.equal(out, rgb)


@pytest.mark.parametrize(("num_tiles", "num_rows", "width"), [(3, 1, 96), (4, 2, 128), (3, 2, 100)])
def test_every_effect_keeps_the_layout_and_changes_pixels(num_tiles: int, num_rows: int, width: int) -> None:
    rgb = _uint8_batch(64, width)
    torch.manual_seed(0)
    every_effect = _only(dict.fromkeys(_EFFECTS, 1.0))
    out = _augment(rgb, every_effect, num_tiles=num_tiles, num_rows=num_rows)
    assert out.shape == rgb.shape
    assert out.dtype == torch.uint8
    assert not torch.equal(out, rgb)


@pytest.mark.parametrize("name", _EFFECTS)
def test_each_effect_runs_alone(name: str) -> None:
    rgb = _uint8_batch(32, 96)
    torch.manual_seed(0)
    out = _augment(rgb, _only({name: 1.0}), num_tiles=3, num_rows=2)
    assert out.shape == rgb.shape


def test_camera_effects_stay_inside_their_tile() -> None:
    rgb = _uint8_batch(32, 96)
    torch.manual_seed(0)
    out = _augment(rgb, _only({"exposure_probability": 0.5}), num_tiles=3, num_rows=1)
    changed_tiles = (out != rgb).view(4, 3, 32, 3, 32).any(dim=(1, 2, 4))
    unchanged = (out == rgb).view(4, 3, 32, 3, 32).all(dim=(1, 2, 4))
    assert torch.equal(changed_tiles, ~unchanged)
    assert changed_tiles.any()
    assert unchanged.any()


def test_past_and_anchor_rows_of_a_camera_share_parameters() -> None:
    row = _uint8_batch(16, 96)
    rgb = torch.cat([row, row], dim=2)
    torch.manual_seed(0)
    out = _augment(rgb, _only({"exposure_probability": 1.0}), num_tiles=3, num_rows=2)
    assert torch.equal(out[:, :, :16], out[:, :, 16:])


def test_same_seed_same_output() -> None:
    rgb = _uint8_batch(32, 96)
    config = PhotometricAugmentationConfig()
    torch.manual_seed(3)
    first = _augment(rgb, config, num_tiles=3, num_rows=1)
    torch.manual_seed(3)
    second = _augment(rgb, config, num_tiles=3, num_rows=1)
    assert torch.equal(first, second)


def test_cap_drops_effects_of_the_same_category() -> None:
    rgb = _uint8_batch(32, 96)
    colour = {"exposure_probability": 1.0, "saturation_probability": 1.0}
    torch.manual_seed(0)
    capped = _augment(rgb, _only(colour), num_tiles=3, num_rows=1)
    torch.manual_seed(0)
    uncapped = _augment(rgb, _only(colour, max_effects_per_category=2), num_tiles=3, num_rows=1)
    assert not torch.equal(capped, uncapped)


def test_clean_samples_stay_untouched() -> None:
    rgb = _uint8_batch(32, 96)
    every_effect = dataclasses.replace(
        _only(dict.fromkeys(_EFFECTS, 1.0), max_effects_per_category=99),
        clean_fraction=1.0,
    )
    assert torch.equal(_augment(rgb, every_effect, num_tiles=3, num_rows=1), rgb)


def test_camera_cap_drops_effects_across_categories() -> None:
    rgb = _uint8_batch(32, 96)
    two = _only({"exposure_probability": 1.0, "blur_probability": 1.0})
    torch.manual_seed(0)
    capped = _augment(rgb, dataclasses.replace(two, max_effects_per_camera=1), num_tiles=3, num_rows=1)
    torch.manual_seed(0)
    uncapped = _augment(rgb, two, num_tiles=3, num_rows=1)
    assert not torch.equal(capped, uncapped)
