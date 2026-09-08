"""Inference-time replacement for Microsoft's mup package (MIT, Copyright (c) 2022
Microsoft Corporation): the MuReadout layers without the width-multiplier scaling,
matching AlpaSim's reference loader."""

from __future__ import annotations

import pickle
from typing import IO, Any, cast

import torch
from torch import nn
from typing_extensions import override


class InfDim:
    def __init__(self, base_dim: float | None, dim: float | None) -> None:
        self.base_dim = base_dim
        self.dim = dim


class InfShape(tuple[InfDim, ...]):
    pass


class MuReadout(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        output_mult: float = 1.0,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)  # pyright: ignore[reportUnknownMemberType]
        self.output_mult = output_mult

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(self.output_mult * input)


class MuSharedReadout(MuReadout):
    def __init__(
        self,
        weight: nn.Parameter,
        bias: bool = False,
        output_mult: float = 1.0,
    ) -> None:
        super().__init__(weight.shape[1], weight.shape[0], bias=bias, output_mult=output_mult)
        self.weight = weight


def normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """mup's width-scaled init; the plain init suffices because loaded weights replace it."""
    return cast("torch.Tensor", tensor.data.normal_(mean=mean, std=std))  # pyright: ignore[reportUnknownMemberType]


def set_base_shapes(
    model: nn.Module,
    base: Any,
    rescale_params: bool = True,
    do_assert: bool = True,
) -> nn.Module:
    """No-op: the width-multiplier bookkeeping is dropped, matching AlpaSim's reference loader."""
    del base, rescale_params, do_assert
    return model


class _CheckpointUnpickler(pickle.Unpickler):
    """Maps the checkpoint's pickled mup shape classes onto the local replacements."""

    @override
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("mup."):
            return {"InfDim": InfDim, "InfShape": InfShape}[name]
        return super().find_class(module, name)


class checkpoint_pickle_module:
    """The pickle_module torch.load needs to unpickle mup shapes without mup installed."""

    Unpickler = _CheckpointUnpickler

    @staticmethod
    def load(file: IO[bytes], **kwargs: Any) -> Any:
        return cast("Any", _CheckpointUnpickler(file, **kwargs).load())
