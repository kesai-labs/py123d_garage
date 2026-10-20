# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Fingerprints of the torch functions the kernel patches replace: a change to one of them fails the unit tests until the
kernels are checked against it on a GPU and the fingerprints are regenerated.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import cast

REGENERATE_COMMAND = "python -m py123d_garage.policy.transfuser.network.torch_path_fingerprints --write"
GPU_TESTS = "pytest py123d_garage/tests/cuda_unit"

FINGERPRINTS: dict[str, str] = {
    "py123d_garage.policy.transfuser.network.decoder.transformer_decoder_layer:PlanningDecoderLayer._ff_block": "a837bc0c0339978eefb7ba925247e892ed46b9b70890d6dd34bcda2b4b8eaa51",
    "py123d_garage.policy.transfuser.network.decoder.transformer_decoder_layer:qk_norm_attention": "97912d581ed9ae3cd05b3fc22456dbaf837293c936056aa250214dcccedd92c5",
    "py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder:AdaptiveNormDecoderLayer.forward": "400cacf64d85ef3c744106251057cd6ea4e9ab84c05b652d8e543b48bb39fcd8",
    "py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder:AdaptiveNormTransformerDecoder.forward": "53ad90d4199a823dac2d7f6794427d2449c8891b56210de846ef9414aee9c29f",
    "py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder:AdaptiveNormTransformerDecoder._fused_forward": "a97a26337384408fdcd7eb973a8fe5c6c69a12a3246c892ba380288ba54cb4a7",
    "py123d_garage.policy.transfuser.network.encoder.transfuser:TransfuserBackbone.fuse_features": "2230c42645ad65cf16be3625d62014676ab0b9c89d9e7bee0c2d254da86bb0f5",
    "py123d_garage.policy.transfuser.network.encoder.transfuser:TransfuserBackbone._latent_lidar_grid": "7025533a9bd5792ee35be8f48e6c2f4ed0e3963833708eeab040172a02c3b3fb",
    "py123d_garage.policy.transfuser.network.encoder.transfuser:_resample_to": "2686e29b8f46dc5208cc941b9e51ddcec097530db0916738937c44e3dcac4a9a",
    "py123d_garage.policy.transfuser.network.encoder.transfuser:SelfAttention.forward": "d87a841c95e6693865220c033277e8ec55115c3bbb60538980fd32371309c9ee",
}


def resolve(target: str) -> types.FunctionType:
    """
    Imports the function a target names.

    Args:
        target: a "module:qualified.name" path.

    Returns:
        the function.
    """
    module_name, _, qualified_name = target.partition(":")
    found = functools.reduce(getattr, qualified_name.split("."), importlib.import_module(module_name))
    if not inspect.isfunction(found):
        raise TypeError(f"{target} is not a function")
    return cast(types.FunctionType, found)


def fingerprint(target: str) -> str:
    """
    Hashes the source text of the function a target names.

    Args:
        target: a "module:qualified.name" path.

    Returns:
        the sha256 hex digest of the source.
    """
    return hashlib.sha256(inspect.getsource(resolve(target)).encode()).hexdigest()


def changed() -> dict[str, tuple[str, str]]:
    """
    Finds the functions whose source no longer matches its fingerprint.

    Returns:
        the recorded and the current fingerprint per changed target.
    """
    current = {target: fingerprint(target) for target in FINGERPRINTS}
    return {
        target: (FINGERPRINTS[target], current[target])
        for target in FINGERPRINTS
        if current[target] != FINGERPRINTS[target]
    }


def write(path: Path, fingerprints: dict[str, str]) -> None:
    """
    Rewrites the table of a copy of this module with the given fingerprints.

    Args:
        path: the module file to rewrite.
        fingerprints: the fingerprint per target, every target of the table.
    """
    lines = path.read_text().splitlines(keepends=True)
    start = lines.index("FINGERPRINTS: dict[str, str] = {\n") + 1
    end = lines.index("}\n", start)
    table = [f'    "{target}": "{fingerprints[target]}",\n' for target in FINGERPRINTS]
    path.write_text("".join(lines[:start] + table + lines[end:]))


def main(arguments: list[str]) -> int:
    """
    Prints the changed functions, or with --write regenerates every fingerprint in this module.

    Args:
        arguments: the command line arguments after the module name.

    Returns:
        0 when nothing changed or the table was written, 1 otherwise.
    """
    if arguments == ["--write"]:
        write(Path(__file__), {target: fingerprint(target) for target in FINGERPRINTS})
        return 0
    differences = changed()
    for target in differences:
        sys.stdout.write(f"{target} changed\n")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
