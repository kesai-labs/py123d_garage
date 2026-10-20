# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The torch functions the CUDA kernels replace still have the source they were validated against; a change here must be
checked against the kernels on a GPU before the fingerprints are regenerated.
"""

from __future__ import annotations

import hashlib
import inspect
import shutil
from pathlib import Path

import pytest

from py123d_garage.policy.transfuser.network import torch_path_fingerprints
from py123d_garage.policy.transfuser.network.torch_path_fingerprints import (
    FINGERPRINTS,
    GPU_TESTS,
    REGENERATE_COMMAND,
    changed,
    fingerprint,
    main,
    resolve,
    write,
)


def test_the_torch_paths_the_kernels_replace_are_unchanged() -> None:
    differences = changed()
    assert not differences, (
        f"{sorted(differences)} replaced by CUDA kernels changed. The kernels were validated against the old code: "
        f"run the GPU tests ({GPU_TESTS}) on a machine with the kernels built, fix the kernels where they no longer "
        f"match, then regenerate the fingerprints with: {REGENERATE_COMMAND}"
    )


def test_every_target_resolves_to_a_function() -> None:
    for target in FINGERPRINTS:
        assert inspect.isfunction(resolve(target)), target


def test_a_fingerprint_follows_the_source_text(monkeypatch: pytest.MonkeyPatch) -> None:
    target = next(iter(FINGERPRINTS))
    recorded = fingerprint(target)
    assert len(recorded) == 64
    monkeypatch.setattr(inspect, "getsource", lambda _function: "def edited(): ...")
    assert fingerprint(target) == hashlib.sha256(b"def edited(): ...").hexdigest()
    assert fingerprint(target) != recorded


def test_write_regenerates_the_table_in_a_copy(tmp_path: Path) -> None:
    copy = tmp_path / "torch_path_fingerprints.py"
    shutil.copy(Path(torch_path_fingerprints.__file__), copy)
    stale = dict.fromkeys(FINGERPRINTS, "0" * 64)
    write(copy, stale)
    assert copy.read_text().count('"' + "0" * 64 + '"') == len(FINGERPRINTS)
    write(copy, {target: fingerprint(target) for target in FINGERPRINTS})
    assert copy.read_text() == Path(torch_path_fingerprints.__file__).read_text()


def test_main_reports_changes_without_writing(capsys: object) -> None:
    assert main([]) == 0
