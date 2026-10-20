# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The checklist of adding a kernel, enforced: every op registered in csrc has a registry entry whose requirements match its
directory, every registry entry has an op, every patch imports, every op has an opcheck example, and the build record
names the architectures.
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser import registry

CSRC = Path(__file__).resolve().parents[3] / "cuda" / "src" / "py123d_garage_cuda" / "transfuser" / "csrc"
DIRECTORY_CONTRACTS: dict[str, tuple[tuple[int, int], bool]] = {
    "sm80": ((8, 0), False),
    "sm90": ((9, 0), False),
    "sm90a": ((9, 0), True),
}


def _schemas() -> dict[str, str]:
    """
    Reads the op names each capability directory's ops.cpp defines.

    Returns:
        the directory name per op.
    """
    directories: dict[str, str] = {}
    for directory in DIRECTORY_CONTRACTS:
        for name in re.findall(r'm\.def\(\s*"(\w+)\(', (CSRC / directory / "ops.cpp").read_text()):
            assert name not in directories, f"{name} is defined twice"
            directories[name] = directory
    return directories


def test_every_csrc_op_is_in_the_registry_and_the_registry_names_only_csrc_ops() -> None:
    csrc_ops = set(_schemas())
    registry_ops = {op for entry in registry.OPS.values() for op in entry.ops}
    assert csrc_ops == registry_ops


def test_every_family_is_named_after_its_forward_op() -> None:
    for family, entry in registry.OPS.items():
        assert entry.ops[0] == family


def test_every_op_lives_in_the_directory_of_its_requirements() -> None:
    for op, directory in _schemas().items():
        entry = registry.OPS[registry.family_of(op)]
        capability, sm90a_only = DIRECTORY_CONTRACTS[directory]
        assert (entry.capability, entry.sm90a_only) == (capability, sm90a_only), f"{op} in {directory}"


def test_every_patch_imports_to_a_function() -> None:
    for patch in registry.PATCHES:
        module, _, function = patch.partition(":")
        assert callable(getattr(importlib.import_module(module), function))


def test_importing_the_package_registers_every_family() -> None:
    script = "import json, sys; import py123d_garage_cuda; print(json.dumps(sorted(m for m in sys.modules if m.startswith('py123d_garage_cuda.'))))"
    imported = set(
        json.loads(subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout),
    )
    needed = {patch.partition(":")[0] for patch, ops in registry.PATCHES.items() if ops}
    assert needed <= imported, sorted(needed - imported)


def test_every_patch_op_is_in_the_registry() -> None:
    for patch, ops in registry.PATCHES.items():
        for op in ops:
            assert op in registry.OPS, f"{patch}: {op}"


def test_every_op_has_an_opcheck_example() -> None:
    from tests.cuda_unit.transfuser.test_opcheck_sweep import EXAMPLES

    assert set(EXAMPLES) == {op for entry in registry.OPS.values() for op in entry.ops}


def test_the_build_record_names_architectures() -> None:
    architectures = registry.built_architectures()
    assert architectures
    for architecture in architectures:
        base = architecture.partition("+")[0].rstrip("a")
        major, minor = (int(part) for part in base.split("."))
        assert (major, minor) >= (8, 0), architecture
