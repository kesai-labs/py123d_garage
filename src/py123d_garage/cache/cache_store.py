# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

"""Reads a built cache store and checks that it matches the caller's config."""

from __future__ import annotations

import ctypes
import logging
import math
import os
import shutil
import typing
import weakref
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from py123d_garage.cache._lmdb_backend import (
    open_log_read_env,
    read_cached_tensor_addresses,
)
from py123d_garage.cache.codec import decode_tensor
from py123d_garage.common.logging_setup import log_stage

if TYPE_CHECKING:
    from concurrent.futures import Executor

    import lmdb
    import torch

LOG = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.yaml"
_DATA_FILE_NAME = "data.mdb"
# Open environments kept per process; reopening one costs ~1.3 ms per read, and 65530 mappings is the kernel limit.
_MAX_OPEN_ENVS = 16_384
# Most chunks one coverage check submits to its process pool; see check_covers.
_MAX_SUBMITS = 256
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_MADV_WILLNEED = 3
_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def _advise_will_need(buffer: bytes | memoryview) -> None:
    """
    Start an asynchronous disk read of the mapped pages under a buffer.

    LMDB maps its stores without readahead, so a cold record otherwise faults in one page at a time.

    Args:
        buffer: a zero-copy view into an LMDB map
    """
    address = np.frombuffer(buffer, dtype=np.uint8).ctypes.data
    start = address - address % _PAGE_SIZE
    _LIBC.madvise(ctypes.c_void_p(start), ctypes.c_size_t(address + len(buffer) - start), _MADV_WILLNEED)


def read_manifest(cache_store_root: str | Path) -> dict[str, Any] | None:
    """
    Read a store's manifest.

    Args:
        cache_store_root: the store's root directory

    Returns:
        the manifest, or None when the store was never built
    """
    import yaml

    manifest_file = Path(cache_store_root) / _MANIFEST_NAME
    if not manifest_file.is_file():
        return None
    return yaml.safe_load(manifest_file.read_text())


def write_manifest(
    cache_store_root: str | Path,
    manifest: dict[str, Any],
) -> None:
    """
    Write the manifest declaring what a store holds, atomically.

    Args:
        cache_store_root: the store's root directory
        manifest: the store's self-description (tensors and cache_signature)
    """
    import yaml

    manifest_file = Path(cache_store_root) / _MANIFEST_NAME
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = manifest_file.with_suffix(f".{os.getpid()}.tmp")
    temporary_file.write_text(yaml.safe_dump(manifest, sort_keys=False))
    temporary_file.replace(manifest_file)


def delete_manifest(cache_store_root: str | Path) -> None:
    """
    Unseal a store so readers refuse it until a finished build reseals it.

    Args:
        cache_store_root: the store's root directory
    """
    (Path(cache_store_root) / _MANIFEST_NAME).unlink(missing_ok=True)


def check_cache_signature(
    manifest: dict[str, Any] | None,
    current: dict[str, dict[str, str]],
) -> None:
    """
    Refuse a store that cannot serve the caller's cache_signature.

    Stored tensors beyond the caller's signature are ignored, and a store without a manifest passes.

    Args:
        manifest: the store's manifest, or None when it was never built
        current: the caller's current cache_signature

    Raises:
        ValueError: if the store was built and cannot serve current
    """
    if manifest is None:
        return
    stored = manifest.get("cache_signature", {})
    changed_tensors = sorted(
        tensor_name for tensor_name, fields in current.items() if stored.get(tensor_name) != fields
    )
    if not changed_tensors:
        return
    details = "\n".join(
        f"  {tensor_name}: "
        + (
            "missing from the store"
            if tensor_name not in stored
            else ", ".join(
                f"{field} {stored[tensor_name].get(field)!r} -> {current[tensor_name].get(field)!r}"
                for field in sorted(
                    stored[tensor_name].keys() | current[tensor_name].keys(),
                )
                if stored[tensor_name].get(field) != current[tensor_name].get(field)
            )
        )
        for tensor_name in changed_tensors
    )
    raise ValueError(
        f"cache store cannot serve the current config, in: {changed_tensors}\n"
        f"{details}\n"
        f"Rebuild the store with force_cache_rebuild=true.",
    )


def check_builder_cache_signature(
    manifest: dict[str, Any] | None,
    current: dict[str, dict[str, str]],
) -> None:
    """
    Refuse to extend a store this build could not reseal faithfully.

    A finishing build reseals the manifest with only its own tensors, so it must not build into a wider store.

    Args:
        manifest: the store's manifest, or None when it was never built
        current: the builder's cache_signature

    Raises:
        ValueError: if the store was built with different settings, or holds
            tensors this build would drop from the manifest
    """
    check_cache_signature(manifest, current)
    if manifest is None:
        return
    extra_tensor_names = sorted(
        manifest.get("cache_signature", {}).keys() - current.keys(),
    )
    if extra_tensor_names:
        raise ValueError(
            f"the store also holds {extra_tensor_names}, which this build does not "
            f"produce; finishing would reseal the manifest without them and refuse "
            f"their consumers. Build into a separate cache_root, or rebuild this "
            f"one with force_cache_rebuild=true.",
        )


class CacheStoreReader:
    """
    Reads prebuilt sample tensors and is safe to hand to forked DataLoader workers.

    LMDB environments open lazily per process; a pickled or forked copy starts without them, since py-lmdb must not use
    an environment after a fork.
    """

    def __init__(
        self,
        cache_store_root: str | Path,
        cache_signature: dict[str, dict[str, str]],
        mirror_root: str | Path | None = None,
    ) -> None:
        """
        Open a built store.

        Args:
            cache_store_root: the store's root directory
            cache_signature: the caller's signature, whose tensors read() serves
            mirror_root: node-local directory each log's data file is copied under on first read; None reads in place

        Raises:
            FileNotFoundError: if the store was never built
        """
        self._cache_store_root = Path(cache_store_root)
        self._mirror_root = Path(mirror_root) if mirror_root is not None else None
        manifest = read_manifest(self._cache_store_root)
        if manifest is None:
            raise FileNotFoundError(
                f"no cache store under {self._cache_store_root}; build it with py123d_garage.cache.build_cache",
            )
        check_cache_signature(manifest, cache_signature)
        self.tensor_names: frozenset[str] = frozenset(
            manifest["tensors"],
        ) & frozenset(cache_signature)
        self._envs: OrderedDict[str, lmdb.Environment] | None = None

    def read(
        self,
        log_name: str,
        sample_key: str,
    ) -> dict[str, torch.Tensor]:
        """
        Read one sample's tensors named in tensor_names.

        Args:
            log_name: the log of the sample
            sample_key: the per-log sample key (anchor timestamp)

        Returns:
            the decoded tensors, one per name in tensor_names

        Raises:
            KeyError: if the store does not hold every tensor for this sample
        """
        tensors: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        with self._env(log_name).begin() as txn:
            for name in self.tensor_names:
                blob = txn.get(f"{sample_key}/{name}".encode())
                if blob is None:
                    missing.append(name)
                    continue
                tensors[name] = decode_tensor(bytes(blob))
        if missing:
            raise KeyError(
                f"{sorted(missing)} of sample {sample_key} of log {log_name} are not in the store "
                f"under {self._cache_store_root}; rebuild it with py123d_garage.cache.build_cache.",
            )
        return tensors

    def prefetch(
        self,
        samples: Sequence[tuple[str, str]],
        executor: Executor,
    ) -> None:
        """
        Ask the kernel to start reading samples' records from disk, so later read() calls find them in the page cache.

        The record lookups run on the executor's threads, which also overlaps the B-tree page reads before each record.

        Args:
            samples: (log name, per-log sample key) of each sample
            executor: the threads doing the lookups
        """
        envs = [self._env(log_name) for log_name, _ in samples]
        list(executor.map(self._advise_sample, envs, [sample_key for _, sample_key in samples]))

    def _advise_sample(self, env: lmdb.Environment, sample_key: str) -> None:
        """
        Look up one sample's records and ask the kernel to read their pages.

        Args:
            env: the sample's open log environment
            sample_key: the per-log sample key
        """
        with env.begin(buffers=True) as txn:
            for name in self.tensor_names:
                blob = txn.get(f"{sample_key}/{name}".encode())
                if blob is not None:
                    _advise_will_need(blob)

    def _env(self, log_name: str) -> lmdb.Environment:
        """
        The log's open environment, opening it and closing the least recently used one past the cap.

        Args:
            log_name: the log

        Returns:
            the open read-only environment
        """
        if self._envs is None:
            self._envs = OrderedDict()
            _OPEN_READERS.add(self)
        env = self._envs.get(log_name)
        if env is None:
            env = open_log_read_env(self.log_dir(log_name))
            self._envs[log_name] = env
            if len(self._envs) > _MAX_OPEN_ENVS:
                self._envs.popitem(last=False)[1].close()
        else:
            self._envs.move_to_end(log_name)
        return env

    def log_dir(self, log_name: str) -> Path:
        """
        Find a log's environment directory, copying it to the mirror root on first use.

        Args:
            log_name: the log to locate

        Returns:
            the log's directory in the store, or its local copy with a mirror root
        """
        log_dir = self._cache_store_root / log_name
        if self._mirror_root is None:
            return log_dir
        mirror_dir = self._mirror_root / log_dir.absolute().relative_to("/")
        data_file = mirror_dir / _DATA_FILE_NAME
        if not data_file.exists():
            import fcntl

            mirror_dir.mkdir(parents=True, exist_ok=True)
            # One worker copies each log under the lock; the rename keeps a half-written copy from ever being opened.
            with (mirror_dir / ".copy.lock").open("w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                if not data_file.exists():
                    partial_file = mirror_dir / f".{_DATA_FILE_NAME}.partial"
                    shutil.copyfile(log_dir / _DATA_FILE_NAME, partial_file)
                    partial_file.replace(data_file)
        return mirror_dir

    def missing_sample_keys(
        self,
        log_name: str,
        sample_keys: Sequence[str],
    ) -> list[str]:
        """
        The given samples the store does not fully hold, from one key-only scan of the log.

        Args:
            log_name: the log to scan
            sample_keys: the per-log sample keys to look for

        Returns:
            the sample keys missing any of tensor_names, in input order
        """
        stored = read_cached_tensor_addresses(self._cache_store_root, log_name)
        return [
            sample_key
            for sample_key in sample_keys
            if any(f"{sample_key}/{tensor_name}" not in stored for tensor_name in self.tensor_names)
        ]

    def check_covers(
        self,
        scene_uuids_by_log: Mapping[str, Sequence[str]],
        executor: Executor | None = None,
    ) -> None:
        """
        Fail fast when the store does not hold every selected scene.

        Args:
            scene_uuids_by_log: the selected scene uuids, grouped by their log
            executor: scans the logs in parallel; None scans them serially

        Raises:
            KeyError: if a log is missing any of its selected scenes
        """
        num_scenes = sum(len(scene_uuids) for scene_uuids in scene_uuids_by_log.values())
        with log_stage(
            f"Checking the store covers the {num_scenes} selected scenes of {len(scene_uuids_by_log)} logs",
        ):
            if executor is None:
                missing_per_log: Iterable[list[str]] = map(
                    self.missing_sample_keys,
                    scene_uuids_by_log.keys(),
                    scene_uuids_by_log.values(),
                )
            else:
                # Thousands of submits can deadlock a Python 3.10 process pool (gh-105829), so cap the chunk count.
                missing_per_log = executor.map(
                    self.missing_sample_keys,
                    scene_uuids_by_log.keys(),
                    scene_uuids_by_log.values(),
                    chunksize=max(16, math.ceil(len(scene_uuids_by_log) / _MAX_SUBMITS)),
                )
            for log_index, ((log_name, scene_uuids), missing) in enumerate(
                zip(scene_uuids_by_log.items(), missing_per_log, strict=True),
            ):
                if (log_index + 1) % 500 == 0:
                    LOG.info(f"Checked {log_index + 1}/{len(scene_uuids_by_log)} logs")
                if missing:
                    raise KeyError(
                        f"{len(missing)} of {len(scene_uuids)} selected scenes "
                        f"of log {log_name} are not in the cache store (e.g. "
                        f"{missing[0]}): the scene filter selects scenes the "
                        f"store was not built with; rerun build_cache.",
                    )

    def close(self) -> None:
        """Release open environments. Idempotent."""
        for env in (self._envs or {}).values():
            env.close()
        self._envs = None

    def __getstate__(self) -> dict[str, typing.Any]:
        """
        Pickle the reader without its open environments.

        Returns:
            the reader's attributes with the environments dropped
        """
        state = self.__dict__.copy()
        state["_envs"] = None
        return state


# The readers of this process with open environments, which a forked child closes before its first read.
_OPEN_READERS: weakref.WeakSet[CacheStoreReader] = weakref.WeakSet()


def _close_inherited_envs() -> None:
    """
    Close the environments a forked child inherited from its parent.

    py-lmdb frees a transaction of an inherited environment without unlinking it from the environment's dependents, so
    every later transaction on it, and closing it, writes through a dangling pointer; the heap corruption surfaces as
    "Attempt to operate on closed/deleted/dropped object" on a live environment or as a crashed worker.
    """
    for reader in list(_OPEN_READERS):
        reader.close()
    _OPEN_READERS.clear()


os.register_at_fork(after_in_child=_close_inherited_envs)
