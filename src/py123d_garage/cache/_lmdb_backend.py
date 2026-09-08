from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)

# Stores built before the manifest existed hold this per-log stamp; scans must skip it.
_LEGACY_META_RECORD_KEY: bytes = b"__meta__"

_DEFAULT_MAP_SIZE_BYTES: int = 1 << 30  # per log, doubled on demand
_MAX_MAP_SIZE_BYTES: int = 1 << 40

_DEFAULT_RECORDS_PER_TRANSACTION: int = 256


def open_log_read_env(log_dir: str | Path, max_readers: int = 2048, readahead: bool = False):
    """
    Open one log's environment for concurrent, lock-free reading.

    Args:
        log_dir: the log's environment directory
        max_readers: LMDB reader-table size of the environment
        readahead: OS readahead; off for random sample access, on for sequential scans

    Returns:
        the open read-only LMDB environment
    """
    import lmdb

    return lmdb.open(
        str(log_dir),
        readonly=True,
        # Keeps the environment out of LMDB's reader table, making it safe to keep
        # using after fork().
        lock=False,
        readahead=readahead,
        max_readers=max_readers,
        subdir=True,
    )


def read_cached_tensor_addresses(
    store_root: str | Path,
    log_name: str,
) -> set[str]:
    """
    Tensor addresses (<sample_key>/<tensor_name>) already stored for a log.

    Args:
        store_root: the store's root directory, laid out as <store_root>/<log_name>/
        log_name: the log to scan

    Returns:
        the stored tensor addresses; empty if the log has no store or its env is damaged
    """
    import lmdb

    log_dir = Path(store_root) / log_name
    if not (log_dir / "data.mdb").is_file():
        return set()
    try:
        env = open_log_read_env(log_dir, readahead=True)
        try:
            with env.begin() as txn:
                return {key.decode() for key in txn.cursor().iternext(values=False) if key != _LEGACY_META_RECORD_KEY}
        finally:
            env.close()
    except lmdb.Error as error:
        LOG.warning(
            f"unreadable store env {log_dir} ({error}); recomputing its log",
        )
        return set()


class LmdbCacheWriter:
    """Exclusive writer for one log's environment. One instance per (store_root, log_name)."""

    def __init__(
        self,
        store_root: str | Path,
        log_name: str,
        map_size: int = _DEFAULT_MAP_SIZE_BYTES,
        records_per_transaction: int = _DEFAULT_RECORDS_PER_TRANSACTION,
    ) -> None:
        """
        Open (creating if needed) one log's environment for writing.

        Args:
            store_root: the store's root directory
            log_name: the log this writer owns
            map_size: initial LMDB address-space reservation in bytes
            records_per_transaction: tensor records batched per write transaction
        """
        import lmdb

        self._lmdb = lmdb
        self._log_dir = Path(store_root) / log_name
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._records_per_transaction = records_per_transaction
        self._map_size = map_size

        self._env = lmdb.open(
            str(self._log_dir),
            map_size=map_size,
            subdir=True,
            # The cache is a derived artifact; durability buys nothing.
            sync=False,
            # A write map would ftruncate data.mdb to the full map_size reservation.
            writemap=False,
        )
        self._pending: list[tuple[bytes, bytes]] = []

    def write(self, tensor_address: str, tensor_blob: bytes) -> int:
        """
        Store one already-encoded sample tensor.

        Args:
            tensor_address: the record key, <sample_key>/<tensor_name>
            tensor_blob: the encoded record

        Returns:
            bytes written, for progress reporting
        """
        self._pending.append((tensor_address.encode(), tensor_blob))
        if len(self._pending) >= self._records_per_transaction:
            self.flush()
        return len(tensor_blob)

    def _commit(self, items: list[tuple[bytes, bytes]]) -> None:
        """
        Write items in one transaction, doubling the map whenever it overflows.

        set_mapsize requires no open transaction; the context manager aborts the failed
        one first.
        """
        assert self._env is not None
        while True:
            try:
                with self._env.begin(write=True) as txn:
                    for key, tensor_blob in items:
                        txn.put(key, tensor_blob)
                return
            except self._lmdb.MapFullError:  # noqa: PERF203
                if self._map_size >= _MAX_MAP_SIZE_BYTES:
                    raise
                self._map_size *= 2
                LOG.info(
                    f"{self._log_dir.name}: growing lmdb map to {self._map_size / (1 << 30):.1f} GiB",
                )
                self._env.set_mapsize(self._map_size)

    def flush(self) -> None:
        """Commit pending records now. Idempotent."""
        if not self._pending:
            return
        self._commit(self._pending)
        self._pending.clear()

    def close(self) -> None:
        """Flush pending writes and release the environment. Idempotent."""
        if self._env is None:
            return
        self.flush()
        self._env.sync(True)
        self._env.close()
        self._env = None

    def __enter__(self) -> LmdbCacheWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
