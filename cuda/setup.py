# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
Builds py123d_garage_cuda._C against the installed torch: pip install --no-build-isolation ./cuda

Compiles for TORCH_CUDA_ARCH_LIST, else the visible GPUs, else Hopper (9.0, built as 9.0a).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _configure_toolchain() -> None:
    """Puts the Python environment's bin on PATH and sets CUDA_HOME from the nvcc found there, unless already set."""
    # Conda-style envs ship nvcc and ninja in their bin, which pip does not put on PATH; a venv on top keeps them in
    # the base prefix.
    env_bins = list(dict.fromkeys(str(Path(prefix) / "bin") for prefix in (sys.prefix, sys.base_prefix)))
    os.environ["PATH"] = os.pathsep.join([*env_bins, os.environ.get("PATH", "")])
    if not os.environ.get("CUDA_HOME") and not os.environ.get("CUDA_PATH"):
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise SystemExit("py123d_garage_cuda needs nvcc: set CUDA_HOME or put nvcc on PATH.")
        os.environ["CUDA_HOME"] = str(Path(nvcc).parent.parent)


_configure_toolchain()

try:
    import torch
    import torch.version
except ImportError:
    raise SystemExit(
        "py123d_garage_cuda compiles against the installed torch, which an isolated build cannot see: "
        "pip install --no-build-isolation ./cuda",
    ) from None

# After _configure_toolchain: cpp_extension reads CUDA_HOME when it is imported.
from setuptools import setup  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

if torch.version.cuda is None:
    raise SystemExit(f"py123d_garage_cuda needs a CUDA build of torch, found {torch.__version__}.")
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    capabilities = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{major}.{minor}" for major, minor in sorted(capabilities)) or "9.0"


def _hopper_as_sm90a(arch: str) -> str:
    """Spells a Hopper entry of TORCH_CUDA_ARCH_LIST as 9.0a, keeping a +PTX suffix: the wgmma kernels need the feature set."""
    base, plus, suffix = arch.partition("+")
    if base in ("9.0", "Hopper"):
        base = "9.0a"
    return base + plus + suffix


os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(
    _hopper_as_sm90a(arch) for arch in os.environ["TORCH_CUDA_ARCH_LIST"].replace(" ", ";").split(";") if arch
)

_PACKAGE = _ROOT / "src" / "py123d_garage_cuda"
(_PACKAGE / "_build.py").write_text(
    "\n".join(
        [
            '"""Written by setup.py: the architectures and libraries this build of py123d_garage_cuda targets."""',
            "",
            "from __future__ import annotations",
            "",
            f"ARCHITECTURES = {tuple(os.environ['TORCH_CUDA_ARCH_LIST'].split(';'))!r}",
            f"TORCH = {torch.__version__!r}",
            f"CUDA = {torch.version.cuda!r}",
            f"CUDNN = {torch.backends.cudnn.version()!r}",
            "",
        ],
    ),
)
_SOURCES = sorted(str(path.relative_to(_ROOT)) for path in _PACKAGE.rglob("*") if path.suffix in (".cpp", ".cu"))


def _cudnn_dirs() -> tuple[str, str]:
    """Finds the include and library directories of the cuDNN torch loads: the nvidia-cudnn wheel, else CUDA_HOME."""
    try:
        import nvidia.cudnn

        root = Path(next(iter(nvidia.cudnn.__path__)))
        return str(root / "include"), str(root / "lib")
    except ImportError:
        home = Path(os.environ.get("CUDNN_HOME") or os.environ.get("CUDA_HOME") or os.environ["CUDA_PATH"])
        return str(home / "include"), str(home / "lib64")


_CUDNN_INCLUDE, _CUDNN_LIB = _cudnn_dirs()

setup(
    ext_modules=[
        CUDAExtension(
            name="py123d_garage_cuda._C",
            sources=_SOURCES,
            include_dirs=[str(_PACKAGE), _CUDNN_INCLUDE],
            # The wheel ships only the versioned library.
            extra_link_args=[f"-L{_CUDNN_LIB}", "-l:libcudnn.so.9", f"-Wl,-rpath,{_CUDNN_LIB}"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "-lineinfo"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
