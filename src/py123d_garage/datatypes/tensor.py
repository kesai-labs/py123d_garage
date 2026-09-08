from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

import torch
from torch import Tensor

BundleT = TypeVar("BundleT", bound="TensorBundle")


@dataclasses.dataclass(frozen=True)
class TensorBundle:
    """
    A frozen dataclass that collates, pins, and moves a collection of tensors from CPU to GPU.

    Subclasses of this class will get the hooks that DataLoader and Lightning look for,
    which a plain dataclass does not provide.
    """

    def named_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, TensorBundle):
                yield from value.named_tensors()
            elif isinstance(value, Tensor):
                yield field.name, value

    def apply(self: BundleT, fn: Callable[[Tensor], Tensor]) -> BundleT:
        updated_fields: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, TensorBundle):
                updated_fields[field.name] = value.apply(fn)
            elif isinstance(value, Tensor):
                updated_fields[field.name] = fn(value)
        return dataclasses.replace(self, **updated_fields)

    def to(self: BundleT, *args: Any, **kwargs: Any) -> BundleT:
        return self.apply(lambda tensor: tensor.to(*args, **kwargs))

    def pin_memory(self: BundleT) -> BundleT:
        return self.apply(lambda tensor: tensor.pin_memory())

    def as_dict(self) -> dict[str, Tensor]:
        return dict(self.named_tensors())

    @classmethod
    def collate(cls: type[BundleT], samples: Sequence[BundleT]) -> BundleT:
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
