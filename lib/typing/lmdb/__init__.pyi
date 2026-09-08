"""Minimal stub for the lmdb package, covering only the APIs this repo uses."""

from collections.abc import Iterator
from types import TracebackType
from typing import Literal, overload

class Error(Exception): ...
class MapFullError(Error): ...

class Cursor:
    def __iter__(self) -> Iterator[tuple[bytes, bytes]]: ...
    @overload
    def iternext(self, keys: bool = ..., values: Literal[True] = ...) -> Iterator[tuple[bytes, bytes]]: ...
    @overload
    def iternext(self, keys: bool = ..., *, values: Literal[False]) -> Iterator[bytes]: ...

class Transaction:
    def __enter__(self) -> Transaction: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def get(self, key: bytes, default: bytes | None = ...) -> bytes | None: ...
    def put(
        self,
        key: bytes,
        value: bytes,
        dupdata: bool = ...,
        overwrite: bool = ...,
        append: bool = ...,
    ) -> bool: ...
    def cursor(self) -> Cursor: ...

class Environment:
    def begin(
        self,
        write: bool = ...,
        buffers: bool = ...,
    ) -> Transaction: ...
    def sync(self, force: bool = ...) -> None: ...
    def set_mapsize(self, map_size: int) -> None: ...
    def close(self) -> None: ...

def open(
    path: str,
    map_size: int = ...,
    subdir: bool = ...,
    readonly: bool = ...,
    metasync: bool = ...,
    sync: bool = ...,
    map_async: bool = ...,
    mode: int = ...,
    create: bool = ...,
    readahead: bool = ...,
    writemap: bool = ...,
    meminit: bool = ...,
    max_readers: int = ...,
    max_dbs: int = ...,
    max_spare_txns: int = ...,
    lock: bool = ...,
) -> Environment: ...
