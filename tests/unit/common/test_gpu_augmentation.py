"""Unit tests for the GPU colour-augmentation pipeline, run on CPU tensors."""

from __future__ import annotations

import pytest
import torch

from py123d_garage.common.sensor import image_augmentation
from py123d_garage.common.sensor.image_augmentation import (
    _elastic_deform,
    augment_rgb_batch,
)


def _uint8_batch(seed: int) -> torch.Tensor:
    """
    Draws a reproducible mid-range uint8 camera batch.

    Args:
        seed: generator seed for the pixel draw.

    Returns:
        a (2, 3, 16, 24) uint8 batch.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        32,
        224,
        (2, 3, 16, 24),
        generator=generator,
        dtype=torch.uint8,
    )


def test_zero_probability_is_exact_identity() -> None:
    rgb = _uint8_batch(seed=0)
    torch.manual_seed(0)
    out = augment_rgb_batch(rgb, probability=0.0)
    assert torch.equal(out, rgb)
    assert out.dtype == torch.uint8


def test_input_batch_is_not_mutated() -> None:
    rgb = _uint8_batch(seed=1)
    pristine = rgb.clone()
    torch.manual_seed(1)
    augment_rgb_batch(rgb, probability=1.0)
    assert torch.equal(rgb, pristine)


def test_enabled_augmentation_preserves_layout_and_changes_pixels() -> None:
    rgb = _uint8_batch(seed=2)
    torch.manual_seed(2)
    out = augment_rgb_batch(rgb, probability=1.0)
    assert out.shape == rgb.shape
    assert out.dtype == torch.uint8
    assert out.device == rgb.device
    for index in range(rgb.shape[0]):
        assert not torch.equal(out[index], rgb[index])
    # Augmented, not destroyed: the image stays close to the original.
    difference = (out.float() - rgb.float()).abs().mean()
    assert difference < 80.0


def test_seeded_runs_are_reproducible() -> None:
    rgb = _uint8_batch(seed=3)
    torch.manual_seed(42)
    first = augment_rgb_batch(rgb, probability=0.7)
    torch.manual_seed(42)
    second = augment_rgb_batch(rgb, probability=0.7)
    assert torch.equal(first, second)

    torch.manual_seed(43)
    third = augment_rgb_batch(rgb, probability=0.7)
    assert not torch.equal(first, third)


def test_empty_batch_passes_through() -> None:
    rgb = torch.zeros(0, 3, 16, 24, dtype=torch.uint8)
    torch.manual_seed(0)
    out = augment_rgb_batch(rgb, probability=1.0)
    assert out.shape == (0, 3, 16, 24)
    assert out.dtype == torch.uint8


def test_elastic_grid_without_displacement_is_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With zero displacement the sampling grid must hit pixel centers exactly."""
    monkeypatch.setattr(image_augmentation, "_ELASTIC_ALPHA", 0.0)
    torch.manual_seed(0)
    x = torch.rand(2, 3, 8, 10) * 255.0
    out = _elastic_deform(x)
    assert torch.allclose(out, x, atol=1e-4)


def test_elastic_deform_moves_mass_at_most_alpha_pixels() -> None:
    """A single bright pixel may travel, but only within the 1-pixel field magnitude."""
    x = torch.zeros(1, 3, 15, 15)
    x[0, :, 7, 7] = 255.0
    torch.manual_seed(0)
    out = _elastic_deform(x)
    bright = (out > 1.0).nonzero()
    assert bright.numel() > 0
    rows, cols = bright[:, 2].float(), bright[:, 3].float()
    assert ((rows - 7.0).abs() <= 2.0).all()
    assert ((cols - 7.0).abs() <= 2.0).all()
