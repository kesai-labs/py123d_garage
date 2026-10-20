# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, TypeVar

import torch
import torch.utils._pytree as pytree
from torch import Tensor

BundleT = TypeVar("BundleT", bound="TensorBundle")


@dataclasses.dataclass(frozen=True)
class TensorBundle:
    """Frozen dataclass of tensors with the collate, pin_memory and to hooks that DataLoader and Lightning look for."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Registers every subclass as a pytree node so torch.func can map over its fields.

        Args:
            **kwargs: passed on to the parent's __init_subclass__.
        """
        super().__init_subclass__(**kwargs)
        # Only tensors and nested bundles are leaves; other fields travel in the context, since vmap only returns tensors.
        pytree.register_pytree_node(
            cls,
            _flatten_bundle,
            lambda values, context: cls(**_unflatten_fields(values, context)),
        )

    def named_tensors(self) -> Iterator[tuple[str, Tensor]]:
        """
        Iterates over the bundle's tensors, including those of nested bundles.

        Yields:
            (field name, tensor) pairs; None and non-tensor fields are skipped.
        """
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, TensorBundle):
                yield from value.named_tensors()
            elif isinstance(value, Tensor):
                yield field.name, value

    def apply(self: BundleT, fn: Callable[[Tensor], Tensor]) -> BundleT:
        """
        Maps a function over every tensor, recursing into nested bundles.

        Args:
            fn: the function applied to each tensor.

        Returns:
            a new bundle with the mapped tensors and the other fields unchanged.
        """
        updated_fields: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, TensorBundle):
                updated_fields[field.name] = value.apply(fn)
            elif isinstance(value, Tensor):
                updated_fields[field.name] = fn(value)
        return dataclasses.replace(self, **updated_fields)

    def to(self: BundleT, *args: Any, **kwargs: Any) -> BundleT:
        """
        Moves or casts every tensor like torch.Tensor.to.

        Args:
            *args: positional arguments of torch.Tensor.to.
            **kwargs: keyword arguments of torch.Tensor.to.

        Returns:
            a new bundle with the moved tensors.
        """
        return self.apply(lambda tensor: tensor.to(*args, **kwargs))

    def pin_memory(self: BundleT) -> BundleT:
        """
        Pins every tensor in page-locked memory for faster host-to-device copies.

        Returns:
            a new bundle with the pinned tensors.
        """
        return self.apply(lambda tensor: tensor.pin_memory())

    def as_dict(self) -> dict[str, Tensor]:
        """
        Collects the bundle's tensors by field name.

        Returns:
            the tensors of named_tensors as a dict.
        """
        return dict(self.named_tensors())

    @classmethod
    def collate(cls: type[BundleT], samples: Sequence[BundleT]) -> BundleT:
        """
        Stacks single-sample bundles into one batched bundle.

        Args:
            samples: the bundles to collate; non-tensor fields are taken from the first.

        Returns:
            the batched bundle with a new leading batch dimension on every tensor.
        """
        collated_fields: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            values = [getattr(sample, field.name) for sample in samples]
            if isinstance(values[0], TensorBundle):
                collated_fields[field.name] = type(values[0]).collate(values)
            elif isinstance(values[0], Tensor):
                collated_fields[field.name] = torch.stack(values)
            else:
                collated_fields[field.name] = values[0]
        return cls(**collated_fields)


def _flatten_bundle(bundle: TensorBundle) -> tuple[list[Any], tuple[list[str], dict[str, Any]]]:
    """
    Splits a bundle into its pytree leaves and the context that rebuilds it.

    Args:
        bundle: the bundle to flatten.

    Returns:
        the tensor and nested-bundle fields, and the leaf names with the remaining fields.
    """
    fields = {field.name: getattr(bundle, field.name) for field in dataclasses.fields(bundle)}
    leaf_names = [name for name, value in fields.items() if isinstance(value, Tensor | TensorBundle)]
    static_fields = {name: value for name, value in fields.items() if name not in leaf_names}
    return [fields[name] for name in leaf_names], (leaf_names, static_fields)


def _unflatten_fields(values: Iterable[Any], context: tuple[list[str], dict[str, Any]]) -> dict[str, Any]:
    """
    Rebuilds a bundle's constructor fields from pytree leaves and their context.

    Args:
        values: the leaves, in the order _flatten_bundle returned them.
        context: the leaf names and remaining fields from _flatten_bundle.

    Returns:
        every field of the bundle by name.
    """
    leaf_names, static_fields = context
    return {**dict(zip(leaf_names, values, strict=True)), **static_fields}
