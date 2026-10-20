# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Photometric augmentation of the stitched training camera batch: real-camera effects and artifacts of rendered views."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import jaxtyping as jt
import torch
from torch import Tensor
from torch.nn import functional as F

_PIXELATE_FACTORS = (2, 3)
_BLUR_SIGMAS = (0.5, 0.75, 1.0)
_MOTION_BLUR_LENGTHS = (2, 3, 4)

# In application order. Per-pixel effects run fused over the whole batch, neighbourhood effects on their samples only.
_SCENE_EFFECTS = ("low_light", "haze", "glare")
_LENS_NEIGHBOURHOOD_EFFECTS = ("bloom", "lens_blobs")
_LENS_EFFECTS = ("vignette", "dark_corners", "chromatic_aberration")
_SOFTNESS_EFFECTS = (
    "render_soft",
    "low_resolution",
    *(f"pixelate_{level}" for level in range(len(_PIXELATE_FACTORS))),
    *(f"blur_{level}" for level in range(len(_BLUR_SIGMAS))),
    *(f"motion_blur_{level}" for level in range(len(_MOTION_BLUR_LENGTHS))),
)
_COLOUR_EFFECTS = ("exposure", "overexposure", "white_balance", "saturation", "gamma", "sensor_noise", "chroma_noise")
_EFFECT_NAMES = (*_SCENE_EFFECTS, *_LENS_NEIGHBOURHOOD_EFFECTS, *_LENS_EFFECTS, *_SOFTNESS_EFFECTS, *_COLOUR_EFFECTS)
_EFFECT_INDEX = {effect_name: index for index, effect_name in enumerate(_EFFECT_NAMES)}


@dataclass(frozen=True)
class PhotometricEffectApplication:
    """The samples a neighbourhood effect applies to and, per sample, the camera tiles it covers."""

    sample_indices: jt.Int[Tensor, " samples"]
    tile_gates: jt.Bool[Tensor, "samples tiles"]
    covers_every_tile: bool


@dataclass(frozen=True)
class PhotometricAugmentationPlan:
    """Which camera tiles of which samples every photometric effect applies to, drawn on the host."""

    effect_tile_gates: jt.Bool[Tensor, "effects batch tiles"]
    neighbourhood_effect_applications: dict[str, PhotometricEffectApplication]
    applies_lens_effects: bool

    def to(self, device: torch.device) -> PhotometricAugmentationPlan:
        """
        Moves the plan to a device through one pinned host buffer and one asynchronous copy, so the host never waits.

        Args:
            device: the target device.

        Returns:
            the plan on the device.
        """
        host_parts = [self.effect_tile_gates.long().flatten()]
        for application in self.neighbourhood_effect_applications.values():
            host_parts += [application.sample_indices, application.tile_gates.long().flatten()]
        host_buffer = torch.cat(host_parts)
        if device.type == "cuda":
            host_buffer = host_buffer.pin_memory()
        device_parts = iter(torch.split(host_buffer.to(device, non_blocking=True), [len(part) for part in host_parts]))
        effect_tile_gates = next(device_parts).view(self.effect_tile_gates.shape).bool()
        neighbourhood_effect_applications = {
            effect_name: PhotometricEffectApplication(
                next(device_parts),
                next(device_parts).view(application.tile_gates.shape).bool(),
                application.covers_every_tile,
            )
            for effect_name, application in self.neighbourhood_effect_applications.items()
        }
        return PhotometricAugmentationPlan(
            effect_tile_gates,
            neighbourhood_effect_applications,
            self.applies_lens_effects,
        )


def augment_camera_batch(
    rgb: jt.Shaped[torch.Tensor, "batch 3 height width"],
    effect_probabilities: Mapping[str, float],
    max_effects_per_category: int,
    max_effects_per_camera: int,
    clean_fraction: float,
    num_tiles: int,
    num_rows: int,
) -> jt.Shaped[torch.Tensor, "batch 3 height width"]:
    """
    Applies photometric effects to a collated [0, 255] camera batch, with the gates drawn on the host so nothing syncs.

    Args:
        rgb: batched channel-first stitched camera images, on any device.
        effect_probabilities: chance of each effect, by effect name (exposure, glare, render_soft, ...).
        max_effects_per_category: per camera, at most this many effects of each category apply.
        max_effects_per_camera: per camera, at most this many effects apply in all, the scene effects aside.
        clean_fraction: chance that a sample is left untouched.
        num_tiles: cameras side by side in a row.
        num_rows: frames stacked top to bottom, the past rows above the anchor row.

    Returns:
        the augmented batch, same dtype, device and layout.
    """
    photometric_augmentation_plan = draw_photometric_augmentation_plan(
        len(rgb),
        num_tiles,
        effect_probabilities,
        max_effects_per_category,
        max_effects_per_camera,
        clean_fraction,
    )
    return apply_photometric_augmentation_plan(rgb, photometric_augmentation_plan.to(rgb.device), num_tiles, num_rows)


def draw_photometric_augmentation_plan(
    batch_size: int,
    num_tiles: int,
    effect_probabilities: Mapping[str, float],
    max_effects_per_category: int,
    max_effects_per_camera: int,
    clean_fraction: float,
) -> PhotometricAugmentationPlan:
    """
    Draws on the host which photometric effects apply to which samples and camera tiles.

    Args:
        batch_size: number of samples.
        num_tiles: cameras side by side in a row.
        effect_probabilities: chance of each effect, by effect name.
        max_effects_per_category: per camera, at most this many effects of each category apply.
        max_effects_per_camera: per camera, at most this many effects apply in all, the scene effects aside.
        clean_fraction: chance that a sample is left untouched.

    Returns:
        the plan, on the host.
    """

    def keep_random(
        gates: jt.Bool[torch.Tensor, "effects batch tiles"],
        count: int,
    ) -> jt.Bool[torch.Tensor, "effects batch tiles"]:
        """
        Keeps a random count of the effects that fire on each camera.

        Args:
            gates: which effects fire on which camera.
            count: most effects kept per camera.

        Returns:
            the kept gates.
        """
        priority = torch.where(gates, torch.rand(gates.shape), -1.0)
        rank = priority.argsort(dim=0, descending=True).argsort(dim=0)
        return gates & (rank < count)

    def capped_gates(names: list[str]) -> dict[str, jt.Bool[torch.Tensor, "batch tiles"]]:
        """
        Draws the per-camera gates of one effect category, capped at max_effects_per_category.

        Args:
            names: the effects of the category.

        Returns:
            the gates by effect name.
        """
        gates = torch.stack([torch.rand(batch_size, num_tiles) < effect_probabilities[name] for name in names])
        return dict(zip(names, keep_random(gates, max_effects_per_category), strict=True))

    augmented = torch.rand(batch_size, 1) >= clean_fraction

    def frame_gate(probability: float) -> jt.Bool[torch.Tensor, "batch tiles"]:
        """
        Draws a gate per frame, shared by all its camera tiles.

        Args:
            probability: chance of the effect per frame.

        Returns:
            the gate of every camera tile.
        """
        return ((torch.rand(batch_size, 1) < probability) & augmented).expand(batch_size, num_tiles)

    gates = {
        **capped_gates(["glare", "bloom", "lens_blobs", "vignette", "dark_corners", "chromatic_aberration"]),
        **capped_gates(["render_soft"]),
        **capped_gates(["low_resolution", "pixelate", "blur", "motion_blur"]),
        **capped_gates(["exposure", "overexposure", "white_balance", "saturation", "gamma"]),
        **capped_gates(["sensor_noise", "chroma_noise"]),
    }
    kept = keep_random(torch.stack(list(gates.values())), max_effects_per_camera) & augmented
    gates = dict(zip(gates, kept, strict=True))
    low_light = frame_gate(effect_probabilities["low_light"])
    haze = frame_gate(effect_probabilities["haze"])
    pixelate_level = torch.randint(0, len(_PIXELATE_FACTORS), (batch_size, num_tiles))
    blur_level = torch.randint(0, len(_BLUR_SIGMAS), (batch_size, num_tiles))
    motion_blur_level = torch.randint(0, len(_MOTION_BLUR_LENGTHS), (batch_size, num_tiles))

    pixelate, blur, motion_blur = gates.pop("pixelate"), gates.pop("blur"), gates.pop("motion_blur")
    gates |= {
        "low_light": low_light,
        "haze": haze,
        "sensor_noise": gates["sensor_noise"] | low_light,
        **{f"pixelate_{level}": pixelate & (pixelate_level == level) for level in range(len(_PIXELATE_FACTORS))},
        **{f"blur_{level}": blur & (blur_level == level) for level in range(len(_BLUR_SIGMAS))},
        **{
            f"motion_blur_{level}": motion_blur & (motion_blur_level == level)
            for level in range(len(_MOTION_BLUR_LENGTHS))
        },
    }
    neighbourhood_effect_applications: dict[str, PhotometricEffectApplication] = {}
    for effect_name in (*_LENS_NEIGHBOURHOOD_EFFECTS, *_SOFTNESS_EFFECTS):
        sample_indices = torch.nonzero(gates[effect_name].any(dim=1)).flatten()
        if len(sample_indices):
            tile_gates = gates[effect_name][sample_indices]
            neighbourhood_effect_applications[effect_name] = PhotometricEffectApplication(
                sample_indices,
                tile_gates,
                bool(tile_gates.all()),
            )
    return PhotometricAugmentationPlan(
        torch.stack([gates[effect_name] for effect_name in _EFFECT_NAMES]),
        neighbourhood_effect_applications,
        any(bool(gates[effect_name].any()) for effect_name in _LENS_EFFECTS),
    )


def apply_photometric_augmentation_plan(
    rgb: jt.Shaped[torch.Tensor, "batch 3 height width"],
    photometric_augmentation_plan: PhotometricAugmentationPlan,
    num_tiles: int,
    num_rows: int,
) -> jt.Shaped[torch.Tensor, "batch 3 height width"]:
    """
    Applies the planned photometric effects; with the plan on the batch's device, nothing waits on the host.

    Args:
        rgb: batched channel-first stitched camera images in [0, 255].
        photometric_augmentation_plan: the drawn plan.
        num_tiles: cameras side by side in a row.
        num_rows: frames stacked top to bottom.

    Returns:
        the augmented batch, same dtype, device and layout.
    """
    compiled = rgb.is_cuda
    effect_tile_gates = photometric_augmentation_plan.effect_tile_gates
    _, _, height, width = rgb.shape
    column_tile, x_in_tile, y_in_row, _, tile_aspect = _tile_geometry(height, width, num_tiles, num_rows, rgb.device)

    def apply_to_samples(
        x: jt.Float[torch.Tensor, "batch 3 height width"],
        effect_name: str,
        effect: Callable[[Tensor], Tensor],
    ) -> jt.Float[torch.Tensor, "batch 3 height width"]:
        """
        Applies a neighbourhood effect to the planned samples and tiles only.

        Args:
            x: the images, modified in place.
            effect_name: name of the effect in the plan.
            effect: the effect on a stack of images.

        Returns:
            the images with the effect applied.
        """
        application = photometric_augmentation_plan.neighbourhood_effect_applications.get(effect_name)
        if application is None:
            return x
        index = application.sample_indices
        selected = x[index]
        if application.covers_every_tile:
            x[index] = effect(selected)
        else:
            mask = application.tile_gates[:, column_tile].view(len(selected), 1, 1, width)
            x[index] = torch.where(mask, effect(selected), selected)
        return x

    def lens_blobs(z: Tensor) -> Tensor:
        """
        Blurs and brightens random round spots, like dirt or water drops on the lens.

        Args:
            z: the images.

        Returns:
            the images with blobs.
        """
        count, num_blobs = len(z), 6
        # Centres in tile coordinates, radii in row heights.
        blob_radius = _blob_columns(0.05 + 0.1 * torch.rand(count, num_tiles, num_blobs, device=z.device), column_tile)
        centre_x = _blob_columns(torch.rand(count, num_tiles, num_blobs, device=z.device), column_tile)
        centre_y = _blob_columns(torch.rand(count, num_tiles, num_blobs, device=z.device), column_tile)
        u = (x_in_tile - centre_x) * tile_aspect / blob_radius
        v = (y_in_row - centre_y) / blob_radius
        blobs = torch.exp(-0.5 * (u**2 + v**2))
        alpha = blobs.amax(dim=1, keepdim=True) * 0.8
        return torch.lerp(z, blur(z, 6.0) * 1.08 + 8.0, alpha)

    def render_soft(z: Tensor) -> Tensor:
        """
        Gives the images the soft, slightly desaturated look of a rendered view.

        Args:
            z: the images.

        Returns:
            the softened images.
        """
        soft = 0.5 * _resize(
            _resize(z, int(height * 0.75), int(width * 0.75), "area"),
            height,
            width,
            "bilinear",
        ) + 0.5 * blur(z, 1.0)
        mean = soft.mean(dim=(1, 2, 3), keepdim=True)
        grey = soft.mean(dim=1, keepdim=True)
        return torch.lerp(z, mean + (grey + (soft - grey) * 0.92 - mean) * 0.93, 0.5)

    x = (_compiled_scene_effects if compiled else _scene_effects)(rgb, effect_tile_gates, num_tiles, num_rows)

    # Lens and windshield.
    x = apply_to_samples(
        x,
        "bloom",
        lambda z: (
            z
            + _resize(
                blur(_resize((z - 170.0).clamp(min=0), height // 4, width // 4, "area"), 3.0),
                height,
                width,
                "bilinear",
            )
            * 0.9
        ),
    )
    x = apply_to_samples(x, "lens_blobs", lens_blobs)
    if photometric_augmentation_plan.applies_lens_effects:
        x = (_compiled_lens_effects if compiled else _lens_effects)(x, effect_tile_gates, num_tiles, num_rows)

    # Rendered views and softness.
    x = apply_to_samples(x, "render_soft", render_soft)
    x = apply_to_samples(
        x,
        "low_resolution",
        lambda z: _resize(_resize(z, int(height * 0.85), int(width * 0.85), "area"), height, width, "bilinear"),
    )
    for level, factor in enumerate(_PIXELATE_FACTORS):
        x = apply_to_samples(
            x,
            f"pixelate_{level}",
            lambda z, factor=factor: _resize(
                _resize(z, max(1, height // factor), max(1, width // factor), "area"),
                height,
                width,
                "nearest",
            ),
        )
    for level, sigma in enumerate(_BLUR_SIGMAS):
        x = apply_to_samples(x, f"blur_{level}", lambda z, sigma=sigma: blur(z, sigma))
    for level, length in enumerate(_MOTION_BLUR_LENGTHS):
        x = apply_to_samples(
            x,
            f"motion_blur_{level}",
            lambda z, length=length: _directional_blur(z, length, horizontal=True),
        )

    return (_compiled_colour_effects if compiled else _colour_effects)(
        x,
        effect_tile_gates,
        num_tiles,
        num_rows,
        rgb.dtype,
    )


def _tile_geometry(
    height: int,
    width: int,
    num_tiles: int,
    num_rows: int,
    device: torch.device,
) -> tuple[
    jt.Int[torch.Tensor, " width"],
    jt.Float[torch.Tensor, "1 1 1 width"],
    jt.Float[torch.Tensor, "1 1 height 1"],
    jt.Float[torch.Tensor, "1 1 height width"],
    float,
]:
    """
    Computes where each pixel sits in its camera tile.

    Args:
        height: image height.
        width: image width.
        num_tiles: cameras side by side in a row.
        num_rows: frames stacked top to bottom.
        device: device of the outputs.

    Returns:
        each column's tile, the pixel's x and y in its tile in [0, 1], its distance from the tile centre, the aspect.
    """
    row_height = height // num_rows
    column_tile = torch.arange(width, device=device) * num_tiles // width
    x_in_tile = ((torch.arange(width, device=device) + 0.5) * num_tiles / width - column_tile).view(1, 1, 1, -1)
    y_in_row = ((torch.arange(height, device=device) % row_height + 0.5) / row_height).view(1, 1, -1, 1)
    radius = torch.sqrt(((x_in_tile - 0.5) / 0.5) ** 2 + ((y_in_row - 0.5) / 0.5) ** 2)
    return column_tile, x_in_tile, y_in_row, radius, width / num_tiles / row_height


def _effect_mask(
    effect_tile_gates: jt.Bool[torch.Tensor, "effects batch tiles"],
    effect_name: str,
    column_tile: jt.Int[torch.Tensor, " width"],
) -> jt.Bool[torch.Tensor, "batch 1 1 width"]:
    """
    Spreads an effect's tile gates over the pixel columns.

    Args:
        effect_tile_gates: the plan's gates of every effect.
        effect_name: the effect.
        column_tile: tile index of each pixel column.

    Returns:
        the pixel columns the effect applies to, per sample.
    """
    gates = effect_tile_gates[_EFFECT_INDEX[effect_name]]
    return gates[:, column_tile].view(len(gates), 1, 1, -1)


def _uniform_columns(
    low: float,
    high: float,
    batch_size: int,
    num_tiles: int,
    column_tile: jt.Int[torch.Tensor, " width"],
    per_tile: bool = True,
) -> jt.Float[torch.Tensor, "batch 1 1 width"]:
    """
    Draws uniform values per sample and camera tile, or per sample, spread over the pixel columns.

    Args:
        low: lower bound of the draw.
        high: upper bound of the draw.
        batch_size: number of samples.
        num_tiles: cameras side by side in a row.
        column_tile: tile index of each pixel column.
        per_tile: draw per camera tile; false shares one draw among a sample's tiles.

    Returns:
        the drawn value of every pixel column.
    """
    values = low + (high - low) * torch.rand(batch_size, num_tiles if per_tile else 1, device=column_tile.device)
    return values.expand(batch_size, num_tiles)[:, column_tile].view(batch_size, 1, 1, -1)


def _blob_columns(
    values: jt.Float[torch.Tensor, "count tiles blobs"],
    column_tile: jt.Int[torch.Tensor, " width"],
) -> jt.Float[torch.Tensor, "count blobs 1 width"]:
    """
    Spreads per-tile blob parameters over the pixel columns.

    Args:
        values: one value per sample, tile and blob.
        column_tile: tile index of each pixel column.

    Returns:
        the value of every blob and pixel column.
    """
    return values[:, column_tile].permute(0, 2, 1).unsqueeze(2)


def _scene_effects(
    rgb: jt.Shaped[torch.Tensor, "batch 3 height width"],
    effect_tile_gates: jt.Bool[torch.Tensor, "effects batch tiles"],
    num_tiles: int,
    num_rows: int,
) -> jt.Float[torch.Tensor, "batch 3 height width"]:
    """
    Applies low light, haze and glare to the whole batch, as float.

    Args:
        rgb: batched channel-first stitched camera images in [0, 255].
        effect_tile_gates: the plan's gates of every effect.
        num_tiles: cameras side by side in a row.
        num_rows: frames stacked top to bottom.

    Returns:
        the images as float32.
    """
    x = rgb.to(torch.float32)
    batch_size, _, height, width = x.shape
    column_tile, x_in_tile, y_in_row, _, tile_aspect = _tile_geometry(height, width, num_tiles, num_rows, x.device)

    def uniform(low: float, high: float, per_tile: bool = True) -> Tensor:
        """
        Draws uniform values spread over the pixel columns.

        Args:
            low: lower bound of the draw.
            high: upper bound of the draw.
            per_tile: draw per camera tile; false shares one draw among a sample's tiles.

        Returns:
            the drawn value of every pixel column.
        """
        return _uniform_columns(low, high, batch_size, num_tiles, column_tile, per_tile)

    low_light = x * uniform(0.75, 0.9, per_tile=False)
    x = torch.where(_effect_mask(effect_tile_gates, "low_light", column_tile), low_light, x)
    haze = _haze(x, uniform(0.085, 0.22, per_tile=False), uniform(170.0, 225.0, per_tile=False), y_in_row)
    x = torch.where(_effect_mask(effect_tile_gates, "haze", column_tile), haze, x)

    gx, gy = uniform(0.1, 0.9), uniform(0.0, 0.35)
    distance = torch.sqrt(((x_in_tile - gx) * tile_aspect) ** 2 + (y_in_row - gy) ** 2)
    flare = 70.0 * torch.exp(-distance / 0.18) + 11.0 * torch.exp(-((distance - 0.45) ** 2) / 0.004)
    ghost_distance = ((x_in_tile + gx - 1) * tile_aspect) ** 2 + (y_in_row + gy - 1) ** 2
    ghost = 19.0 * torch.exp(-ghost_distance / 0.003)
    channel = torch.arange(3, device=x.device).view(1, 3, 1, 1)
    tint = torch.where(channel == 0, 1.0, torch.where(channel == 1, 0.97, 0.95))
    glare = x + (flare + ghost) * tint
    return torch.where(_effect_mask(effect_tile_gates, "glare", column_tile), glare, x)


def _lens_effects(
    x: jt.Float[torch.Tensor, "batch 3 height width"],
    effect_tile_gates: jt.Bool[torch.Tensor, "effects batch tiles"],
    num_tiles: int,
    num_rows: int,
) -> jt.Float[torch.Tensor, "batch 3 height width"]:
    """
    Applies vignette, dark corners and chromatic aberration to the whole batch.

    Args:
        x: the images in [0, 255].
        effect_tile_gates: the plan's gates of every effect.
        num_tiles: cameras side by side in a row.
        num_rows: frames stacked top to bottom.

    Returns:
        the images with the lens effects.
    """
    batch_size, _, height, width = x.shape
    column_tile, _, _, radius, _ = _tile_geometry(height, width, num_tiles, num_rows, x.device)

    def uniform(low: float, high: float) -> Tensor:
        """
        Draws uniform values per camera tile, spread over the pixel columns.

        Args:
            low: lower bound of the draw.
            high: upper bound of the draw.

        Returns:
            the drawn value of every pixel column.
        """
        return _uniform_columns(low, high, batch_size, num_tiles, column_tile)

    vignette = x * cast(Tensor, 1 - uniform(0.13, 0.32) * (radius / math.sqrt(2)).clamp(0, 1) ** 2)
    x = torch.where(_effect_mask(effect_tile_gates, "vignette", column_tile), vignette, x)
    dark_corners = x * cast(Tensor, 1 - torch.sigmoid((radius - uniform(1.2, 1.4)) * 25.0))
    x = torch.where(_effect_mask(effect_tile_gates, "dark_corners", column_tile), dark_corners, x)
    red = F.pad(x[:, 0:1], (0, 1, 0, 0), mode="replicate")[..., 1:]
    blue = F.pad(x[:, 2:3], (1, 0, 0, 0), mode="replicate")[..., :width]
    chromatic_aberration = torch.cat([red, x[:, 1:2], blue], dim=1)
    return torch.where(_effect_mask(effect_tile_gates, "chromatic_aberration", column_tile), chromatic_aberration, x)


def _colour_effects(
    x: jt.Float[torch.Tensor, "batch 3 height width"],
    effect_tile_gates: jt.Bool[torch.Tensor, "effects batch tiles"],
    num_tiles: int,
    num_rows: int,
    dtype: torch.dtype,
) -> jt.Shaped[torch.Tensor, "batch 3 height width"]:
    """
    Applies exposure, colour and sensor noise to the whole batch, then rounds to [0, 255] in the given dtype.

    Args:
        x: the images in [0, 255].
        effect_tile_gates: the plan's gates of every effect.
        num_tiles: cameras side by side in a row.
        num_rows: frames stacked top to bottom.
        dtype: dtype of the output.

    Returns:
        the augmented images.
    """
    batch_size, _, height, width = x.shape
    column_tile, _, _, _, _ = _tile_geometry(height, width, num_tiles, num_rows, x.device)

    def uniform(low: float, high: float) -> Tensor:
        """
        Draws uniform values per camera tile, spread over the pixel columns.

        Args:
            low: lower bound of the draw.
            high: upper bound of the draw.

        Returns:
            the drawn value of every pixel column.
        """
        return _uniform_columns(low, high, batch_size, num_tiles, column_tile)

    def mask(effect_name: str) -> Tensor:
        """
        Looks up the pixel columns an effect applies to.

        Args:
            effect_name: the effect.

        Returns:
            the column mask per sample.
        """
        return _effect_mask(effect_tile_gates, effect_name, column_tile)

    x = torch.where(mask("exposure"), x * uniform(0.82, 1.2), x)
    x = torch.where(mask("overexposure"), x * uniform(1.08, 1.2) + 3.0, x)

    # Street-light casts at night, mild in daylight.
    strength = torch.where(x.mean(dim=(1, 2, 3)) < 50.0, 0.26, 0.17).view(-1, 1, 1)
    gains = 1 + (2 * torch.rand(batch_size, num_tiles, 3, device=x.device) - 1) * strength
    x = torch.where(mask("white_balance"), x * gains[:, column_tile].permute(0, 2, 1).unsqueeze(2), x)

    grey = x.mean(dim=1, keepdim=True)
    x = torch.where(mask("saturation"), grey + (x - grey) * uniform(0.3, 1.5), x)
    x = torch.where(mask("gamma"), 255.0 * (x.clamp(0, 255) / 255.0) ** uniform(0.8, 1.25), x)

    # Shot noise grows with brightness, read noise is constant; much stronger in low light.
    low_light = effect_tile_gates[_EFFECT_INDEX["low_light"]]
    noise_scale = torch.where(low_light, 1.3, 1.0) * (0.3 + 0.6 * torch.rand(batch_size, num_tiles, device=x.device))
    shot = torch.randn_like(x) * torch.sqrt(x.clamp(min=0) * 0.6)
    read = torch.randn_like(x) * 4.0
    noise_columns = noise_scale[:, column_tile].view(batch_size, 1, 1, width)
    x = torch.where(mask("sensor_noise"), x + (shot + read) * noise_columns, x)
    chroma_noise = _resize(
        torch.randn(batch_size, 3, height // 4, width // 4, device=x.device) * 7.0,
        height,
        width,
        "bilinear",
    )
    x = torch.where(mask("chroma_noise"), x + chroma_noise, x)

    return x.round().clamp(0.0, 255.0).to(dtype)


_compiled_scene_effects = torch.compile(_scene_effects, fullgraph=True, dynamic=False)  # pyright: ignore[reportUnknownMemberType]
_compiled_lens_effects = torch.compile(_lens_effects, fullgraph=True, dynamic=False)  # pyright: ignore[reportUnknownMemberType]
_compiled_colour_effects = torch.compile(_colour_effects, fullgraph=True, dynamic=False)  # pyright: ignore[reportUnknownMemberType]


def _haze(
    x: jt.Float[torch.Tensor, "count 3 height width"],
    amount: jt.Float[torch.Tensor, "count 1 1 width"],
    airlight: jt.Float[torch.Tensor, "count 1 1 width"],
    y_in_row: jt.Float[torch.Tensor, "1 1 height 1"],
) -> jt.Float[torch.Tensor, "count 3 height width"]:
    """
    Blends the images towards a bright airlight, strongest around the horizon of each row.

    Args:
        x: the images.
        amount: the blend weight at the horizon.
        airlight: brightness of the haze.
        y_in_row: the row coordinate of each pixel row in [0, 1].

    Returns:
        the hazy images.
    """
    weight = amount * torch.exp(-((y_in_row - 0.45) ** 2) / 0.08)
    return torch.lerp(x, airlight.expand_as(x), weight)


def _resize(
    x: jt.Float[torch.Tensor, "count channels in_height in_width"],
    height: int,
    width: int,
    mode: str,
) -> jt.Float[torch.Tensor, "count channels height width"]:
    """
    Resizes the images.

    Args:
        x: the images.
        height: the target height.
        width: the target width.
        mode: the torch interpolation mode.

    Returns:
        the resized images.
    """
    return cast(Tensor, F.interpolate(x, size=(height, width), mode=mode))  # pyright: ignore[reportUnknownMemberType]


def blur(
    x: jt.Float[torch.Tensor, "count channels height width"],
    sigma: float,
) -> jt.Float[torch.Tensor, "count channels height width"]:
    """
    Applies a separable gaussian blur with replicate padding.

    Args:
        x: the images.
        sigma: the standard deviation in pixels.

    Returns:
        the blurred images.
    """
    reach = max(1, math.ceil(3 * sigma))
    offsets = torch.arange(-reach, reach + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    x = F.conv2d(
        F.pad(x, (reach, reach, 0, 0), mode="replicate"),
        kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
        groups=channels,
    )
    return F.conv2d(
        F.pad(x, (0, 0, reach, reach), mode="replicate"),
        kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
        groups=channels,
    )


def _directional_blur(
    x: jt.Float[torch.Tensor, "count channels height width"],
    length: int,
    horizontal: bool,
) -> jt.Float[torch.Tensor, "count channels height width"]:
    """
    Applies a box blur along one axis with replicate padding, a streak like motion during the exposure.

    Args:
        x: the images.
        length: the streak length in pixels.
        horizontal: streak along the width, else along the height.

    Returns:
        the blurred images.
    """
    channels = x.shape[1]
    kernel = torch.full((length,), 1.0 / length, device=x.device, dtype=x.dtype)
    before, after = length // 2, length - 1 - length // 2
    if horizontal:
        padded = F.pad(x, (before, after, 0, 0), mode="replicate")
        return F.conv2d(padded, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    padded = F.pad(x, (0, 0, before, after), mode="replicate")
    return F.conv2d(padded, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)
