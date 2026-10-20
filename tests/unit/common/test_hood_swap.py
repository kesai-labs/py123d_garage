# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Unit tests for the hood swap on the front camera, run on CPU tensors."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from py123d_garage.common.sensor import hood_swap
from py123d_garage.common.sensor.hood_swap import (
    _ego_hood,  # pyright: ignore[reportPrivateUsage]
    ego_hood_names,
    paste_hood,
    swap_hoods,
    vehicle_type_from_mount,
)


def _uint8_batch(batch_size: int, height: int, width: int, seed: int = 0) -> torch.Tensor:
    """
    Draws a reproducible mid-range uint8 camera batch.

    Args:
        batch_size: number of samples.
        height: image height.
        width: image width.
        seed: generator seed for the pixel draw.

    Returns:
        a (batch_size, 3, height, width) uint8 batch.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(32, 224, (batch_size, 3, height, width), generator=generator, dtype=torch.uint8)


def _donors() -> dict[str, list[str]]:
    """The donors table of the asset."""
    return json.loads(hood_swap._PAI_HOOD_MASKS_FILE.read_text())["donors"]


def test_zero_probability_is_exact_identity() -> None:
    rgb = _uint8_batch(4, 256, 352)
    out = swap_hoods(rgb, torch.tensor([0, 3, -1, 9]), 0.0, 0.3, num_tiles=1, num_rows=1, front_tile=0)
    assert torch.equal(out, rgb)


def test_only_the_bottom_of_the_front_tile_changes() -> None:
    rgb = _uint8_batch(4, 256, 1056)
    torch.manual_seed(0)
    # Type 0 takes the other type 0, type 7 an AlpaSim hood, the vehicle of no known type any hood.
    out = swap_hoods(rgb, torch.tensor([0, 0, 7, -1]), 1.0, 0.3, num_tiles=3, num_rows=1, front_tile=1)
    changed = out != rgb
    assert out.dtype == torch.uint8
    assert not changed[..., :352].any()
    assert not changed[..., 704:].any()
    assert not changed[:, :, :128].any()
    assert changed[..., 352:704].flatten(1).any(dim=1).all()


def test_a_hood_without_a_covering_donor_stays() -> None:
    # Only type 4 covers type 4, and no other sample is type 4.
    assert _donors()["4"] == ["4"]
    rgb = _uint8_batch(3, 256, 352)
    torch.manual_seed(0)
    out = swap_hoods(rgb, torch.tensor([4, 0, -1]), 1.0, 0.3, num_tiles=1, num_rows=1, front_tile=0)
    assert torch.equal(out[0], rgb[0])


def test_past_and_anchor_rows_take_the_same_donor() -> None:
    row = _uint8_batch(4, 128, 352)
    rgb = torch.cat([row, row], dim=2)
    torch.manual_seed(0)
    out = swap_hoods(rgb, torch.tensor([0, 2, 7, -1]), 1.0, 0.3, num_tiles=1, num_rows=2, front_tile=0)
    assert torch.equal(out[:, :, :128], out[:, :, 128:])


def test_every_type_keeps_its_own_hood_as_donor_and_no_hood_takes_thin_ones() -> None:
    donors = _donors()
    for vehicle_type, type_donors in donors.items():
        if vehicle_type != "none":
            assert vehicle_type in type_donors
    assert "hyperion_8_1" in donors["none"]
    assert "hyperion_8" not in donors["none"]


def test_a_mount_maps_to_its_nearest_type_and_an_unknown_one_fails() -> None:
    centroids = hood_swap._pai_mount_centroids()
    assert vehicle_type_from_mount(list(centroids[5])) == 5
    with pytest.raises(ValueError, match="no known type"):
        vehicle_type_from_mount([0.0] * 9)


def test_ego_hood_names_hold_alpasim_rigs() -> None:
    assert {"hyperion_8", "hyperion_8_1"} <= set(ego_hood_names())


@pytest.mark.parametrize("rendered_hood", [None, "hyperion_8_1"])
def test_paste_hood_replaces_only_the_hood_region(rendered_hood: str | None) -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(60, 200, size=(540, 960, 3), dtype=np.uint8)
    pasted = paste_hood(frame, "hyperion_8", rendered_hood)
    assert pasted.shape == frame.shape
    assert pasted.dtype == np.uint8
    # The sky half is far from any hood and its margin.
    np.testing.assert_array_equal(pasted[:200], frame[:200])
    # Where the overlay is opaque, the frame shows the overlay scaled to the frame's brightness.
    _, alpha = _ego_hood("hyperion_8", 540, 960)
    colour, _ = _ego_hood("hyperion_8", 540, 960)
    opaque = (alpha > 0.99).numpy()
    gain = float(np.clip(frame.mean() / 110.0, 0.6, 1.6))
    expected = colour.permute(1, 2, 0).numpy()[opaque] * gain
    assert np.abs(pasted[opaque].astype(np.float32) - expected).mean() < 8.0


def test_paste_hood_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="no ego hood"):
        paste_hood(np.zeros((540, 960, 3), dtype=np.uint8), "no_such_hood", None)
