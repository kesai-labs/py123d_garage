# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Reads the ptxas -v lines of a kernel build and compares every kernel's register spills with the checked-in baseline:
no kernel may spill more bytes than recorded, and no kernel that did not spill may start to. `update` rewrites the
baseline from a build log.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[2] / "cuda" / "ptxas_spills.json"
ENTRY = re.compile(r"Compiling entry function '(\S+)' for '(sm_\d+a?)'")
SPILL = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")


def demangle(names: list[str]) -> list[str]:
    """
    Demangles kernel names, which folds the per-build hash of an anonymous namespace into "(anonymous namespace)".

    Args:
        names: mangled names.

    Returns:
        the demangled names, or the mangled ones without a demangler on PATH.
    """
    filt = shutil.which("c++filt") or shutil.which("x86_64-conda-linux-gnu-c++filt")
    if filt is None:
        return names
    out = subprocess.run([filt], input="\n".join(names) + "\n", capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def spills_of(log: Path) -> dict[str, dict[str, list[int]]]:
    """
    Collects the spill bytes of every kernel that spills, per architecture.

    Args:
        log: the build log with ptxas -v output.

    Returns:
        {architecture: {demangled kernel: [spill store bytes, spill load bytes]}}.
    """
    records: list[tuple[str, str, int, int]] = []
    current: tuple[str, str] | None = None
    for line in log.read_text().splitlines():
        if entry := ENTRY.search(line):
            current = (entry.group(1), entry.group(2))
        elif (spill := SPILL.search(line)) and current is not None:
            stores, loads = int(spill.group(1)), int(spill.group(2))
            if stores or loads:
                records.append((*current, stores, loads))
            current = None
    names = demangle([name for name, _, _, _ in records])
    result: dict[str, dict[str, list[int]]] = {}
    for name, (_, arch, stores, loads) in zip(names, records, strict=True):
        result.setdefault(arch, {})[name] = [stores, loads]
    return {arch: dict(sorted(kernels.items())) for arch, kernels in sorted(result.items())}


def main() -> None:
    """Runs `check <log>` or `update <log>`."""
    command, log = sys.argv[1], Path(sys.argv[2])
    current = spills_of(log)
    if command == "update":
        BASELINE.write_text(json.dumps(current, indent=2) + "\n")
        print(f"{sum(len(kernels) for kernels in current.values())} spilling kernels recorded in {BASELINE}")
        return
    baseline: dict[str, dict[str, list[int]]] = json.loads(BASELINE.read_text())
    regressions: list[str] = []
    improvements: list[str] = []
    for arch, kernels in current.items():
        if arch not in baseline:
            regressions.append(f"{arch}: no baseline; run `ptxas_spills.py update` on a build for it")
            continue
        for name, (stores, loads) in kernels.items():
            recorded = baseline[arch].get(name)
            if recorded is None:
                regressions.append(
                    f"{arch}: {name} spills {stores} store / {loads} load bytes and did not spill before",
                )
            elif stores > recorded[0] or loads > recorded[1]:
                regressions.append(
                    f"{arch}: {name} spills {stores} / {loads} bytes, baseline {recorded[0]} / {recorded[1]}",
                )
            elif stores < recorded[0] or loads < recorded[1]:
                improvements.append(
                    f"{arch}: {name} spills {stores} / {loads} bytes, baseline {recorded[0]} / {recorded[1]}",
                )
    for arch, kernels in baseline.items():
        if arch in current:
            improvements.extend(f"{arch}: {name} no longer spills" for name in set(kernels) - set(current[arch]))
    for line in improvements:
        print("better than the baseline:", line)
    for line in regressions:
        print("regression:", line)
    print(f"{sum(len(kernels) for kernels in current.values())} spilling kernels, {len(regressions)} regressions")
    if regressions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
