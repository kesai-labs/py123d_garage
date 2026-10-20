# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Inference-time stand-in for Microsoft's mup package (MIT, Copyright (c) 2022 Microsoft Corporation).

The readout layers drop the width-multiplier scaling, matching AlpaSim's reference loader.
"""

from __future__ import annotations

import pickle
from typing import IO, Any, cast

import torch
from torch import nn
from typing_extensions import override


class InfDim:
    """Stand-in for mup's InfDim so pickled checkpoint shapes load."""

    def __init__(self, base_dim: float | None, dim: float | None) -> None:
        """
        Stores the base and actual width of one dimension.

        Args:
            base_dim: width of the dimension in the base model.
            dim: width of the dimension in this model.
        """
        self.base_dim = base_dim
        self.dim = dim


class InfShape(tuple[InfDim, ...]):
    """Stand-in for mup's InfShape so pickled checkpoint shapes load."""


class MuReadout(nn.Linear):
    """Linear output layer that scales its input by a constant multiplier."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        output_mult: float = 1.0,
    ) -> None:
        """
        Creates the layer.

        Args:
            in_features: input width.
            out_features: output width.
            bias: whether the layer has a bias.
            output_mult: multiplier applied to the input.
        """
        super().__init__(in_features, out_features, bias=bias)  # pyright: ignore[reportUnknownMemberType]
        self.output_mult = output_mult

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Applies the scaled linear projection.

        Args:
            input: features with in_features as the last dimension.

        Returns:
            the projected features with out_features as the last dimension.
        """
        return super().forward(self.output_mult * input)


class MuSharedReadout(MuReadout):
    """MuReadout whose weight is shared with an embedding table."""

    def __init__(
        self,
        weight: nn.Parameter,
        bias: bool = False,
        output_mult: float = 1.0,
    ) -> None:
        """
        Creates the layer around a shared weight.

        Args:
            weight: shared weight shaped [out_features, in_features].
            bias: whether the layer has a bias.
            output_mult: multiplier applied to the input.
        """
        super().__init__(weight.shape[1], weight.shape[0], bias=bias, output_mult=output_mult)
        self.weight = weight


def normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """
    Fills the tensor with a plain normal init, enough because loaded weights replace it.

    Args:
        tensor: tensor to fill in place.
        mean: mean of the normal distribution.
        std: standard deviation of the normal distribution.

    Returns:
        the filled tensor.
    """
    return cast("torch.Tensor", tensor.data.normal_(mean=mean, std=std))  # pyright: ignore[reportUnknownMemberType]


def set_base_shapes(
    model: nn.Module,
    base: Any,
    rescale_params: bool = True,
    do_assert: bool = True,
) -> nn.Module:
    """
    Does nothing, since the width-multiplier bookkeeping is dropped.

    Args:
        model: model to return unchanged.
        base: unused base shapes.
        rescale_params: unused.
        do_assert: unused.

    Returns:
        the model, unchanged.
    """
    del base, rescale_params, do_assert
    return model


class _CheckpointUnpickler(pickle.Unpickler):
    """Maps the checkpoint's pickled mup shape classes onto the local replacements."""

    @override
    def find_class(self, module: str, name: str) -> Any:
        """
        Resolves mup classes to the local stand-ins and everything else as usual.

        Args:
            module: module name stored in the pickle.
            name: class name stored in the pickle.

        Returns:
            the class to instantiate.
        """
        if module.startswith("mup."):
            return {"InfDim": InfDim, "InfShape": InfShape}[name]
        return super().find_class(module, name)


class checkpoint_pickle_module:
    """The pickle_module torch.load needs to unpickle mup shapes without mup installed."""

    Unpickler = _CheckpointUnpickler

    @staticmethod
    def load(file: IO[bytes], **kwargs: Any) -> Any:
        """
        Unpickles one object with the mup-aware unpickler.

        Args:
            file: binary file to read.
            **kwargs: extra arguments for the unpickler.

        Returns:
            the unpickled object.
        """
        return cast("Any", _CheckpointUnpickler(file, **kwargs).load())
