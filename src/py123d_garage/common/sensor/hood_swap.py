# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Swaps the own hood on the front camera of a training batch for a covering one, and pastes ego hoods in evaluation."""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F

from py123d_garage.common.sensor.photometric_augmentation import blur

_ASSETS_DIR = Path(__file__).parent / "assets"
# Ego-hood overlays, RGBA at 960 x 540: AlpaSim's rigs and the physical-ai-av vehicle types (pai_<type>).
_EGO_HOODS_DIR = _ASSETS_DIR / "ego_hoods"
# AlpaSim's ego-hood overlays (nre-artifacts/ego-hoods/<rig>/camera_front_wide_120fov.png, Apache-2.0).
_ALPASIM_RIGS = ("hyperion_8", "hyperion_8_1")
_ALPASIM_HOOD_FILES = tuple(_EGO_HOODS_DIR / f"{rig}.png" for rig in _ALPASIM_RIGS)
# Per physical-ai-av vehicle type: the mount centroid and the hand-drawn own-vehicle polygons on the front camera.
_PAI_HOOD_MASKS_FILE = _ASSETS_DIR / "pai_hood_masks.json"
# A mount farther than this from every centroid is a vehicle of no known type.
_MAX_MOUNT_DISTANCE_M = 0.3
# Mean brightness of a daylight frame; an AlpaSim hood scales with the frame's own brightness against it.
_DAYLIGHT_BRIGHTNESS = 110.0
# In pixels of a 256-row frame, scaled to the row height: the blur over the covered region, the light blur after the
# paste, the margin the cover reaches beyond the hoods, and how far inside its edge a donor hood is taken.
_COVER_BLUR_SIGMA = 14.0
_SOFTEN_SIGMA = 1.2
_COVER_REACH = 12
_DONOR_INSET = 4


def vehicle_type_from_mount(mount_m: Sequence[float]) -> int:
    """
    Finds the physical-ai-av vehicle type of a camera mount.

    Args:
        mount_m: the camera_to_imu translations of the front, left and right cameras (ftcam_f0, ftcam_l0, ftcam_r0),
            nine numbers in metres.

    Returns:
        the type of the nearest mount centroid.

    Raises:
        ValueError: when no centroid is near, a vehicle of no known type.
    """
    centroids = _pai_mount_centroids()
    distances = np.linalg.norm(centroids - np.asarray(mount_m, dtype=np.float64), axis=1)
    nearest = int(distances.argmin())
    if distances[nearest] > _MAX_MOUNT_DISTANCE_M:
        raise ValueError(
            f"camera mount {list(mount_m)} is {distances[nearest]:.2f} m from the nearest vehicle type, more than "
            f"{_MAX_MOUNT_DISTANCE_M} m: a vehicle of no known type",
        )
    return nearest


def swap_hoods(
    rgb: jt.Shaped[torch.Tensor, "batch 3 height width"],
    vehicle_types: jt.Int[torch.Tensor, " batch"],
    probability: float,
    alpasim_probability: float,
    num_tiles: int,
    num_rows: int,
    front_tile: int,
) -> jt.Shaped[torch.Tensor, "batch 3 height width"]:
    """
    Swaps the own hood on the front tile of a random share of the samples for a covering hood of the batch or AlpaSim.

    Args:
        rgb: batched channel-first stitched camera images in [0, 255], on any device.
        vehicle_types: the physical-ai-av vehicle type of each sample, -1 for none known.
        probability: chance per sample of a swap.
        alpasim_probability: chance that a swap takes an AlpaSim hood when one qualifies.
        num_tiles: cameras side by side in a row.
        num_rows: frames stacked top to bottom.
        front_tile: the tile of the front camera.

    Returns:
        the batch with hoods swapped, same dtype, device and layout.
    """
    batch_size, _, height, width = rgb.shape
    device = rgb.device
    columns = torch.nonzero(torch.arange(width) * num_tiles // width == front_tile).flatten()
    start, stop = int(columns[0]), int(columns[-1]) + 1
    tile_width = stop - start
    row_height = height // num_rows
    tables = _hood_tables(row_height, tile_width)
    cover_masks, paste_alphas, alpasim_colours, covers = (table.to(device) for table in tables)
    num_pai_types = len(cover_masks) - len(_ALPASIM_HOOD_FILES) - 1
    no_hood = len(cover_masks) - 1

    # Donor choice: a random qualifying sample of the batch and a random qualifying AlpaSim hood per sample.
    own_type = torch.where(vehicle_types >= 0, vehicle_types, no_hood)
    batch_qualifies = covers[vehicle_types.clamp(min=0)][:, own_type].T
    batch_qualifies &= (vehicle_types >= 0).unsqueeze(0) & ~torch.eye(batch_size, dtype=torch.bool, device=device)
    alpasim_qualifies = covers[num_pai_types : num_pai_types + len(_ALPASIM_HOOD_FILES)][:, own_type].T
    batch_donor = torch.where(batch_qualifies, torch.rand(batch_qualifies.shape, device=device), -1.0).argmax(dim=1)
    alpasim_donor = torch.where(alpasim_qualifies, torch.rand(alpasim_qualifies.shape, device=device), -1.0).argmax(1)
    any_batch, any_alpasim = batch_qualifies.any(dim=1), alpasim_qualifies.any(dim=1)
    use_alpasim = any_alpasim & ((torch.rand(batch_size, device=device) < alpasim_probability) | ~any_batch)
    swap = (torch.rand(batch_size, device=device) < probability) & (any_batch | any_alpasim)
    donor_type = torch.where(use_alpasim, num_pai_types + alpasim_donor, vehicle_types[batch_donor].clamp(min=0))

    tiles = rgb[..., start:stop].float().reshape(batch_size, 3, num_rows, row_height, tile_width).transpose(1, 2)
    batch_colour = tiles[batch_donor]
    alpasim_colour = alpasim_colours[alpasim_donor].unsqueeze(1).expand_as(batch_colour)
    tiles, batch_colour, alpasim_colour = (
        x.reshape(batch_size * num_rows, 3, row_height, tile_width) for x in (tiles, batch_colour, alpasim_colour)
    )

    def per_row(x: torch.Tensor) -> torch.Tensor:
        """
        Repeats a per-sample tensor once per row, shaped to broadcast over the tiles.

        Args:
            x: one value per sample.

        Returns:
            one value per row of every sample.
        """
        return x.repeat_interleave(num_rows, dim=0).view(-1, *([1] * (tiles.dim() - 1)))

    tile_brightness = tiles.mean(dim=(1, 2, 3))
    donor_brightness = torch.where(
        per_row(use_alpasim).flatten(),
        _DAYLIGHT_BRIGHTNESS,
        batch_colour.mean(dim=(1, 2, 3)),
    )
    gain = (tile_brightness / donor_brightness.clamp(min=1.0)).clamp(0.6, 1.6).view(-1, 1, 1, 1)
    colour = torch.where(per_row(use_alpasim), alpasim_colour, batch_colour) * gain
    paste_alpha = paste_alphas[donor_type].repeat_interleave(num_rows, dim=0).unsqueeze(1)

    scale = row_height / 256
    reach = max(1, round(_COVER_REACH * scale))
    covered = torch.maximum(cover_masks[own_type], cover_masks[donor_type]).repeat_interleave(num_rows, dim=0)
    cover = blur(F.max_pool2d(covered.unsqueeze(1), kernel_size=2 * reach + 1, stride=1, padding=reach), reach / 2)
    swapped = torch.lerp(tiles, blur(tiles, _COVER_BLUR_SIGMA * scale), cover)
    swapped = torch.lerp(swapped, colour, paste_alpha)
    swapped = torch.lerp(swapped, blur(swapped, _SOFTEN_SIGMA * scale), cover)
    swapped = torch.where(per_row(swap), swapped, tiles)

    swapped = swapped.view(batch_size, num_rows, 3, row_height, tile_width).transpose(1, 2)
    out = rgb.clone()
    out[..., start:stop] = swapped.reshape(batch_size, 3, height, tile_width).round().clamp(0, 255).to(rgb.dtype)
    return out


def ego_hood_names() -> list[str]:
    """
    Lists the overlays paste_hood can paste.

    Returns:
        AlpaSim's rigs and the physical-ai-av vehicle types (pai_<type>), sorted.
    """
    return sorted(path.stem for path in _EGO_HOODS_DIR.glob("*.png"))


@torch.no_grad()
def paste_hood(
    rgb: jt.UInt8[npt.NDArray[np.uint8], "height width 3"],
    hood: str,
    rendered_hood: str | None,
    device: torch.device | str = "cpu",
) -> jt.UInt8[npt.NDArray[np.uint8], "height width 3"]:
    """
    Replaces the hood of one front camera frame with an ego-hood overlay, the way swap_hoods does in training.

    Args:
        rgb: the front camera frame in [0, 255].
        hood: the overlay to paste, one of ego_hood_names().
        rendered_hood: the overlay the frame already shows, blurred away under the new one; None for a frame without a
            hood.
        device: where the blurs run.

    Returns:
        the frame with the hood replaced, same shape and dtype.
    """
    height, width = rgb.shape[:2]
    colour, alpha = (x.to(device) for x in _ego_hood(hood, height, width))
    covered = alpha > 0.5
    if rendered_hood is not None:
        covered |= _ego_hood(rendered_hood, height, width)[1].to(device) > 0.5
    scale = height / 256
    reach = max(1, round(_COVER_REACH * scale))
    dilated = F.max_pool2d(covered.float()[None, None], kernel_size=2 * reach + 1, stride=1, padding=reach)
    cover = blur(dilated, reach / 2)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).permute(2, 0, 1).float().unsqueeze(0)  # pyright: ignore[reportUnknownMemberType]
    gain = (image.mean() / _DAYLIGHT_BRIGHTNESS).clamp(0.6, 1.6)
    swapped = torch.lerp(image, blur(image, _COVER_BLUR_SIGMA * scale), cover)
    swapped = torch.lerp(swapped, colour.unsqueeze(0) * gain, alpha[None, None])
    swapped = torch.lerp(swapped, blur(swapped, _SOFTEN_SIGMA * scale), cover)
    out = swapped.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu()
    return cast("npt.NDArray[np.uint8]", out.numpy())  # pyright: ignore[reportUnknownMemberType]


@functools.cache
def _ego_hood(
    hood: str,
    height: int,
    width: int,
) -> tuple[jt.Float[torch.Tensor, "3 height width"], jt.Float[torch.Tensor, "height width"]]:
    """
    Loads an ego-hood overlay at a frame's size.

    Args:
        hood: the overlay, one of ego_hood_names().
        height: frame height.
        width: frame width.

    Returns:
        its colour in [0, 255] and its alpha in [0, 1].

    Raises:
        ValueError: for a name with no overlay.
    """
    hood_file = _EGO_HOODS_DIR / f"{hood}.png"
    if not hood_file.is_file():
        raise ValueError(f"no ego hood {hood!r}; the hoods are {ego_hood_names()}")
    bgra = cast("npt.NDArray[np.uint8]", cv2.imread(str(hood_file), cv2.IMREAD_UNCHANGED))
    resized = cast("npt.NDArray[np.uint8]", cv2.resize(bgra, (width, height), interpolation=cv2.INTER_AREA))
    rgba = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGRA2RGBA).astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
    return rgba[..., :3].permute(2, 0, 1).contiguous(), rgba[..., 3] / 255.0


@functools.cache
def _pai_mount_centroids() -> jt.Float64[npt.NDArray[np.float64], "types 9"]:
    """
    Loads the mount centroid of each physical-ai-av vehicle type.

    Returns:
        the centroids, indexed by vehicle type.
    """
    centroids = cast("dict[str, list[float]]", json.loads(_PAI_HOOD_MASKS_FILE.read_text())["mount_centroids"])
    return np.array([centroids[str(vehicle_type)] for vehicle_type in range(len(centroids))], dtype=np.float64)


@functools.cache
def _hood_tables(
    height: int,
    width: int,
) -> tuple[
    jt.Float[torch.Tensor, "hoods height width"],
    jt.Float[torch.Tensor, "hoods height width"],
    jt.Float[torch.Tensor, "alpasim_hoods 3 height width"],
    jt.Bool[torch.Tensor, "hoods hoods"],
]:
    """
    Builds the hoods at a camera tile's size: the physical-ai-av vehicle types, then AlpaSim's hoods, then no hood.

    Args:
        height: tile height.
        width: tile width.

    Returns:
        each hood's region to cover, the alpha its own pixels are pasted with, the AlpaSim hoods' colours in
        [0, 255], and whether hood i may replace own hood j.
    """
    polygons = cast("dict[str, list[list[list[float]]]]", json.loads(_PAI_HOOD_MASKS_FILE.read_text())["masks"])
    regions: list[npt.NDArray[np.float32]] = []
    for vehicle_type in range(len(polygons)):
        region = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons[str(vehicle_type)]:
            points = np.round(np.asarray(polygon) * [width - 1, height - 1]).astype(np.int32)
            cv2.fillPoly(region, [points], 1)
        regions.append(region.astype(np.float32))
    colours: list[npt.NDArray[np.float32]] = []
    for hood_file in _ALPASIM_HOOD_FILES:
        bgra = cast("npt.NDArray[np.uint8]", cv2.imread(str(hood_file), cv2.IMREAD_UNCHANGED))
        resized = cast("npt.NDArray[np.uint8]", cv2.resize(bgra, (width, height), interpolation=cv2.INTER_AREA))
        rgba = cast("npt.NDArray[np.uint8]", cv2.cvtColor(resized, cv2.COLOR_BGRA2RGBA)).astype(np.float32)
        colours.append(rgba[..., :3].transpose(2, 0, 1))
        regions.append((rgba[..., 3] > 127).astype(np.float32))
    regions.append(np.zeros((height, width), dtype=np.float32))
    cover_masks = torch.from_numpy(np.stack(regions))  # pyright: ignore[reportUnknownMemberType]

    scale = height / 256
    inset = max(1, round(_DONOR_INSET * scale))
    # A donor's own pixels from inside its edge, so no road next to it comes along.
    eroded = -F.max_pool2d(-cover_masks.unsqueeze(1), kernel_size=2 * inset + 1, stride=1, padding=inset)
    paste_alphas = blur(eroded, max(0.5, scale)).squeeze(1)
    num_pai_types = len(polygons)
    for index in range(len(_ALPASIM_HOOD_FILES)):
        bgra = cast("npt.NDArray[np.uint8]", cv2.imread(str(_ALPASIM_HOOD_FILES[index]), cv2.IMREAD_UNCHANGED))
        resized = cast("npt.NDArray[np.uint8]", cv2.resize(bgra, (width, height), interpolation=cv2.INTER_AREA))
        paste_alphas[num_pai_types + index] = torch.from_numpy(resized[..., 3].astype(np.float32) / 255.0)  # pyright: ignore[reportUnknownMemberType]

    # covers[i, j]: the donors table lets hood i replace own hood j.
    names = [*(str(vehicle_type) for vehicle_type in range(len(polygons))), *_ALPASIM_RIGS, "none"]
    donors = cast("dict[str, list[str]]", json.loads(_PAI_HOOD_MASKS_FILE.read_text())["donors"])
    covers = torch.zeros(len(names), len(names), dtype=torch.bool)
    for own, own_donors in donors.items():
        for donor in own_donors:
            covers[names.index(donor), names.index(own)] = True
    return cover_masks, paste_alphas, torch.from_numpy(np.stack(colours)), covers  # pyright: ignore[reportUnknownMemberType]
