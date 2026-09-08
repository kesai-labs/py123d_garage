import enum
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

from typing_extensions import Self

_T = TypeVar("_T")

class classproperty(Generic[_T]):
    f: Callable[[Any], _T]
    def __init__(self, f: Callable[[Any], _T]) -> None: ...
    def __get__(self, obj: object, owner: type | None = ...) -> _T: ...

class SerialIntEnum(enum.Enum):
    def __int__(self) -> int: ...
    def serialize(self, lower: bool = True) -> str: ...
    @classmethod
    def deserialize(cls, key: str) -> Self: ...
    @classmethod
    def from_int(cls, value: int) -> Self: ...
    @classmethod
    def from_arbitrary(cls, value: int | str | SerialIntEnum) -> Self: ...

def resolve_enum_arguments(
    serial_enum_cls: type[SerialIntEnum],
    input: Sequence[int | str | SerialIntEnum] | None,
) -> list[SerialIntEnum] | None: ...
