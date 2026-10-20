"""Minimal stub for the diskcache package, covering only the APIs this repo uses.

diskcache ships no type information; values are pickled, so get and set are typed on Any.
"""

from typing import Any

class FanoutCache:
    def __init__(
        self,
        directory: str | None = None,
        shards: int = 8,
        timeout: float = 0.01,
        *,
        size_limit: int = ...,
        cull_limit: int = ...,
        sqlite_journal_mode: str = ...,
    ) -> None: ...
    def get(self, key: str, default: Any = None, retry: bool = False) -> Any: ...
    def set(self, key: str, value: Any, retry: bool = False) -> bool: ...
    def __contains__(self, key: str) -> bool: ...
    def __len__(self) -> int: ...
    def close(self) -> None: ...
