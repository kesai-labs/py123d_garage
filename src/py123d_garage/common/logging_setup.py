from __future__ import annotations

import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from typing_extensions import override

try:
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    rank_zero_only = None

LOG = logging.getLogger(__name__)

# log_stage's frame plus contextlib's __enter__/__exit__ frame.
_CALLER_STACKLEVEL = 3


class _RepoPathFormatter(logging.Formatter):
    """Names a record's source by its path in the repo, third-party ones by file."""

    def __init__(self, fmt: str) -> None:
        super().__init__(fmt)
        self._working_dir = Path.cwd()

    @override
    def format(self, record: logging.LogRecord) -> str:
        absolute_source_file = record.pathname
        try:
            record.pathname = str(
                Path(absolute_source_file).relative_to(self._working_dir),
            )
        except ValueError:  # Outside the repo, e.g. a library's own logs.
            record.pathname = record.filename
        try:
            return super().format(record)
        finally:
            record.pathname = absolute_source_file


class _RankFilter(logging.Filter):
    """Drops INFO and below from every rank but zero, and tags what survives."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        # rank_zero_only.rank is attached by lightning and rank_tag by this
        # filter, so neither is visible to their declared types.
        rank: int = getattr(rank_zero_only, "rank", 0)
        record.__dict__["rank_tag"] = "" if rank == 0 else f"[rank {rank}] "
        return rank == 0 or record.levelno >= logging.WARNING


def setup_logging() -> None:
    """Routes INFO logs to stdout; without it the root logger drops them."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    already_on_stdout = any(getattr(handler, "stream", None) is sys.stdout for handler in root.handlers)
    if not already_on_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            _RepoPathFormatter(
                "[%(asctime)s][%(levelname)s]%(rank_tag)s[%(pathname)s:%(lineno)d] %(message)s",
            ),
        )
        # On the handler, so it applies to records propagating up from any
        # logger, not only those logged through the root one.
        handler.addFilter(_RankFilter())
        root.addHandler(handler)
    for existing in root.manager.loggerDict.values():
        if isinstance(existing, logging.Logger):
            existing.disabled = False


@contextmanager
def log_stage(description: str) -> Generator[None]:
    """
    Logs a stage's start and its wall time when it ends.

    The startup stages read the whole dataset's metadata and take minutes;
    without both ends logged a run is indistinguishable from a hung one.

    Args:
        description: what the stage does, e.g. "Listing logs of nuplan_train".

    Yields:
        nothing; the stage runs inside the context.
    """
    # Past this generator and the contextmanager frame that drives it, so the
    # records carry the `with` statement's file and line, not this one's.
    LOG.info(f"{description} ...", stacklevel=_CALLER_STACKLEVEL)
    start_time = time.perf_counter()
    yield
    LOG.info(
        f"{description}: done in {time.perf_counter() - start_time:.1f} s",
        stacklevel=_CALLER_STACKLEVEL,
    )
