# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Valeo
# Adapted from valeoai/VideoActionModel (MIT License).

"""Image transforms that turn camera frames into VaVAM tokenizer inputs."""

from typing import Tuple

import numpy as np
import torch

import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from torch import Tensor


class TopCrop:
    """Crops a fixed number of rows off the top of an image."""

    def __init__(self, trop_crop_size: int) -> None:
        """
        Stores the crop size.

        Args:
            trop_crop_size: number of rows to remove from the top.
        """
        self.trop_crop_size = trop_crop_size

    def __call__(self, img: Tensor) -> Tensor:
        """
        Crops the image.

        Args:
            img: image shaped [channels, height, width].

        Returns:
            the image without its top rows.
        """
        return TF.crop(img, self.trop_crop_size, 0, img.shape[1] - self.trop_crop_size, img.shape[2])


class ResizeByFactor:
    """Downsizes an image by a constant factor."""

    def __init__(self, resize_factor: float) -> None:
        """
        Stores the factor.

        Args:
            resize_factor: divisor applied to the height and width.
        """
        self.resize_factor = resize_factor

    def __call__(self, img: Tensor) -> Tensor:
        """
        Resizes the image.

        Args:
            img: image shaped [channels, height, width].

        Returns:
            the resized image.
        """
        new_width = int(img.shape[2] / self.resize_factor)
        new_height = int(img.shape[1] / self.resize_factor)
        return TF.resize(img, (new_height, new_width), antialias=True)


class Normalize:
    """Maps an image from [0, 1] to [-1, 1]."""

    def __call__(self, img: Tensor) -> Tensor:
        """
        Normalizes the image.

        Args:
            img: image with values in [0, 1].

        Returns:
            the image with values in [-1, 1].
        """
        return 2 * img - 1


class CropAndResizeTransform:
    """Top crop, downsize and normalization to [-1, 1] in one transform."""

    def __init__(self, trop_crop_size: int, resize_factor: float) -> None:
        """
        Builds the transform chain.

        Args:
            trop_crop_size: number of rows to remove from the top.
            resize_factor: divisor applied to the height and width.
        """
        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                TopCrop(trop_crop_size),
                ResizeByFactor(resize_factor),
                transforms.ToDtype(torch.float32, scale=True),
                Normalize(),
            ],
        )

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Applies the transform chain.

        Args:
            *args: inputs of the torchvision Compose.
            **kwargs: keyword inputs of the torchvision Compose.

        Returns:
            the transformed image.
        """
        return self.transforms(*args, **kwargs)


def torch_image_to_plot(img: Tensor, to_numpy: bool = True) -> np.ndarray:
    """
    Converts normalized images back to uint8 for plotting.

    Args:
        img: images in [-1, 1] shaped [batch, channels, height, width].
        to_numpy: whether to return a numpy array instead of a tensor.

    Returns:
        uint8 images shaped [batch, height, width, channels].
    """
    img = torch.clamp(127.5 * img + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8)
    if to_numpy:
        img = img.numpy()
    return img


class SafeResize:
    """Snaps an image to the expected size, then downsizes it by a constant factor."""

    def __init__(self, resize_factor: float, size: Tuple[int, int]) -> None:
        """
        Stores the factor and the expected size.

        Args:
            resize_factor: divisor applied to the height and width.
            size: expected (height, width) before downsizing.
        """
        self.resize_factor = resize_factor
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        """
        Resizes the image.

        Args:
            img: image shaped [channels, height, width].

        Returns:
            the image at size divided by resize_factor.
        """
        height, width = img.shape[1], img.shape[2]
        if (height != self.size[0]) or (width != self.size[1]):
            img = TF.resize(img, self.size, antialias=True)

        new_width = int(img.shape[2] / self.resize_factor)
        new_height = int(img.shape[1] / self.resize_factor)
        return TF.resize(img, (new_height, new_width), antialias=True)


class NeuroNCAPTransform:
    """
    Resizes frames to the nuScenes size, downsizes them and normalizes to [-1, 1].

    The first resize absorbs inputs a pixel off the nuScenes size, e.g. (900, 1599) instead of (900, 1600).
    """

    def __init__(self, resize_factor: float = 3.125, default_size: Tuple[int, int] = (900, 1600)) -> None:
        """
        Builds the transform chain.

        Args:
            resize_factor: divisor applied to the height and width after the first resize.
            default_size: (height, width) every input is first resized to.
        """
        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                SafeResize(resize_factor, default_size),
                transforms.ToDtype(torch.float32, scale=True),
                Normalize(),
            ],
        )

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Applies the transform chain.

        Args:
            *args: inputs of the torchvision Compose, e.g. one HWC uint8 image.
            **kwargs: keyword inputs of the torchvision Compose.

        Returns:
            the image shaped [3, height, width] with values in [-1, 1].
        """
        return self.transforms(*args, **kwargs)
