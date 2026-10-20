# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Writes the clang compile database of the kernel sources: what torch's extension builder passes to nvcc, spelled for clang."""

from __future__ import annotations

import json
import os
import shutil
import sys
import sysconfig
from pathlib import Path

import nvidia.cudnn
from torch.utils import cpp_extension

PACKAGE = Path(__file__).resolve().parents[2] / "cuda" / "src" / "py123d_garage_cuda"
DEFINES = (
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    "-DTORCH_EXTENSION_NAME=_C",
)


def main() -> None:
    """Writes compile_commands.json into the directory given as the first argument, for the GPU architecture given as the second."""
    out_dir, arch = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "sm_90a"
    clang = shutil.which("clang++")
    if clang is None:
        raise SystemExit("clang++ is not on PATH")
    cudnn_include = Path(next(iter(nvidia.cudnn.__path__))) / "include"
    system_includes = [*cpp_extension.include_paths(), sysconfig.get_paths()["include"], str(cudnn_include)]
    common = [clang, "-std=c++17", "-O3", "-fPIC", f"-I{PACKAGE}"]
    for include in system_includes:
        common += ["-isystem", include]
    common += DEFINES
    cuda_only = [
        "-x",
        "cuda",
        "--cuda-device-only",
        f"--cuda-path={os.environ['CUDA_HOME']}",
        f"--cuda-gpu-arch={arch}",
        "-Wno-unknown-cuda-version",
        "-D_CG_CLUSTER_INTRINSICS_AVAILABLE",
    ]
    entries = []
    for source in sorted([*PACKAGE.rglob("*.cu"), *PACKAGE.rglob("*.cpp")]):
        arguments = [*common, *(cuda_only if source.suffix == ".cu" else []), "-c", str(source), "-o", "/dev/null"]
        entries.append({"directory": str(PACKAGE), "file": str(source), "arguments": arguments})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compile_commands.json").write_text(json.dumps(entries, indent=1))
    print(f"{len(entries)} translation units in {out_dir / 'compile_commands.json'}")


if __name__ == "__main__":
    main()
