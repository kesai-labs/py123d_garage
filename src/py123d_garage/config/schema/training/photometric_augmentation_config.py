# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field, fields

from py123d_garage.datatypes.numerics import NonNegativeFloat, PositiveInt


@dataclass
class PhotometricAugmentationConfig:
    """Chance of each photometric effect on the training camera batch, per camera tile or per frame."""

    @property
    def effect_probabilities(self) -> dict[str, float]:
        """The chance of each effect by effect name: the *_probability fields without the suffix."""
        return {
            config_field.name.removesuffix("_probability"): getattr(self, config_field.name)
            for config_field in fields(self)
            if config_field.name.endswith("_probability")
        }

    # Most effects of one category (lens, render, softness, colour, sensor) on one camera.
    max_effects_per_category: PositiveInt = 2
    # Most effects on one camera in all, the scene effects aside.
    max_effects_per_camera: PositiveInt = 2
    # Chance that a sample gets no effect at all.
    clean_fraction: NonNegativeFloat = 0.25

    # Chance of a darker scene, once per frame.
    low_light_probability: NonNegativeFloat = 0.13
    # Chance of haze over the scene, once per frame.
    haze_probability: NonNegativeFloat = 0.13
    # Chance of a brighter or darker exposure, per camera since each camera exposes on its own.
    exposure_probability: NonNegativeFloat = 0.28
    # Chance of a white balance shift, per camera.
    white_balance_probability: NonNegativeFloat = 0.28
    # Chance of a saturation change, per camera.
    saturation_probability: NonNegativeFloat = 0.28
    # Chance of a gamma change, per camera.
    gamma_probability: NonNegativeFloat = 0.21
    # Chance of blown-out highlights, per camera.
    overexposure_probability: NonNegativeFloat = 0.04
    # Chance of a down- and up-scaled, softer image, per camera.
    low_resolution_probability: NonNegativeFloat = 0.14
    # Chance of blocky 2x or 3x pixels, as a low-bitrate stream shows, per camera.
    pixelate_probability: NonNegativeFloat = 0.08
    # Chance of a blur, per camera.
    blur_probability: NonNegativeFloat = 0.21
    # Chance of a motion blur, per camera.
    motion_blur_probability: NonNegativeFloat = 0.21
    # Chance of brightness noise, per camera.
    sensor_noise_probability: NonNegativeFloat = 0.28
    # Chance of colour noise, per camera.
    chroma_noise_probability: NonNegativeFloat = 0.13
    # Chance of a vignette, per camera.
    vignette_probability: NonNegativeFloat = 0.18
    # Chance of the dark corners of a wide-angle lens, which the Physical-AI-AV and NuRec front cameras show.
    dark_corners_probability: NonNegativeFloat = 0.18
    # Chance of colour fringes at edges, per camera.
    chromatic_aberration_probability: NonNegativeFloat = 0.07
    # Chance of light bleeding around bright areas, per camera.
    bloom_probability: NonNegativeFloat = 0.24
    # Chance of a sun flare with lens ghosts, once per frame.
    glare_probability: NonNegativeFloat = 0.11
    # Chance of blurry spots from dirt or water drops on the lens, per camera.
    lens_blobs_probability: NonNegativeFloat = 0.13
    # Chance of the soft look of a rendered view (Gaussian splatting, neural reconstruction), per camera.
    render_soft_probability: NonNegativeFloat = 0.42


@dataclass
class HoodSwapConfig:
    """Swapping the own vehicle's hood on the front camera for a covering one of the batch or AlpaSim's."""

    # Chance per sample that its front camera's hood is swapped.
    probability: NonNegativeFloat = 0.5
    # Chance that a swap takes one of AlpaSim's hoods instead of another sample's, when one qualifies.
    alpasim_probability: NonNegativeFloat = 0.3
    # Datasets whose vehicle type is read from the camera mounts; a log that matches no type fails.
    typed_datasets: list[str] = field(default_factory=lambda: ["physical-ai-av", "nurec"])
    # Datasets whose front camera shows no hood; they get thin donor hoods, and any other dataset fails.
    hoodless_datasets: list[str] = field(default_factory=lambda: ["nuplan", "carla"])


@dataclass
class CameraAugmentationConfig:
    """Augmentation of the camera batch during training; each policy hands over its camera layout."""

    # Apply the photometric effects: real camera conditions and artifacts of rendered views.
    photometric_augmentation: bool = False
    # Chances of the photometric effects.
    photometric_augmentation_config: PhotometricAugmentationConfig = field(
        default_factory=PhotometricAugmentationConfig,
    )
    # Swap the front camera's hood for another one, so the hood tells nothing about the vehicle; runs first.
    hood_swap: bool = False
    # Settings of the hood swap.
    hood_swap_config: HoodSwapConfig = field(default_factory=HoodSwapConfig)
