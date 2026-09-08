from __future__ import annotations

import logging
import os
import typing
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    """Unseal a store: readers refuse it until a finished build reseals it. Idempotent."""
    (Path(cache_store_root) / _MANIFEST_NAME).unlink(missing_ok=True)


def check_cache_signature(
    manifest: dict[str, Any] | None,
    current: dict[str, dict[str, str]],
) -> None:
    """
    Refuse a store that cannot serve the caller's cache_signature.

    Every tensor of the caller's signature must be stored with the same
    signature; tensors the store holds beyond those are ignored, so a store
    built with more heads enabled serves any consumer of a subset of it. A
    store with no manifest yet (never built) is not a mismatch; the caller is
    about to build it.

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

    The reader check accepts a store holding more tensors than the caller
    needs, but a finishing builder reseals the manifest with only its own
    tensors: building into a wider store would shrink its declaration and
    refuse the wider consumers it still serves.

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
    Reads prebuilt sample tensors. Safe to hand to forked DataLoader workers.

    LMDB environments are opened lazily per process and stripped when pickled, since
    neither the memory map nor the file descriptor survives a fork.
    """

    def __init__(
        self,
        cache_store_root: str | Path,
        cache_signature: dict[str, dict[str, str]],
    ) -> None:
        """
        Open a built store.

        Args:
            cache_store_root: the store's root directory
            cache_signature: the caller's signature, checked against the store's;
                its tensors are what read() serves — stored tensors beyond it
                are left untouched

        Raises:
            FileNotFoundError: if the store was never built
        """
        self._cache_store_root = Path(cache_store_root)
        manifest = read_manifest(self._cache_store_root)
        if manifest is None:
            raise FileNotFoundError(
                f"no cache store under {self._cache_store_root}; build it with py123d_garage.cache.build_cache",
            )
        check_cache_signature(manifest, cache_signature)
        self.tensor_names: frozenset[str] = frozenset(
            manifest["tensors"],
        ) & frozenset(cache_signature)
        self._envs: dict[str, lmdb.Environment] | None = None

    def read(
        self,
        log_name: str,
        sample_key: str,
    ) -> dict[str, torch.Tensor]:
        """
        Read one sample's tensor_names — the stored tensors the signature declares.

        Args:
            log_name: the log of the sample
            sample_key: the per-log sample key (anchor timestamp)

        Returns:
            the decoded tensors, one per name in tensor_names

        Raises:
            KeyError: if the store does not hold every tensor for this sample
        """
        if self._envs is None:
            self._envs = {}
        env = self._envs.get(log_name)
        if env is None:
            env = open_log_read_env(self._cache_store_root / log_name)
            self._envs[log_name] = env
        tensors: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        with env.begin() as txn:
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
        Fails fast when the store does not hold every selected scene.

        Grouping the scenes by log stays with the caller: training walks a lazy
        scene sequence, evaluation a materialized list.

        Args:
            scene_uuids_by_log: the selected scene uuids, grouped by their log.
            executor: scans the logs in parallel; None scans them serially.

        Raises:
            KeyError: if a log is missing any of its selected scenes.
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
                missing_per_log = executor.map(
                    self.missing_sample_keys,
                    scene_uuids_by_log.keys(),
                    scene_uuids_by_log.values(),
                    chunksize=16,
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
        state = self.__dict__.copy()
        state["_envs"] = None
        return state
