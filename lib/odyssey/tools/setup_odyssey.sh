#!/bin/bash

set -euo pipefail

ODYSSEY_COMMIT="8c4a902ebfe242feb624d00e756c6c1ccc95555e"

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
_runtime="${PWD}/lib/odyssey/runtime"
_conda="$(command -v micromamba || command -v conda)"
export TMPDIR="${_runtime}/tmp"
mkdir -p "${TMPDIR}"
# A conda env's activation adds its own -ccbin here, and nvcc refuses a second one.
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS

# 1. Clone the Odyssey repo, and nvdiffrast into it
if [ ! -d lib/odyssey/odyssey/.git ]; then
  git clone https://github.com/SNU-ADR/Odyssey.git lib/odyssey/odyssey
fi
cd lib/odyssey/odyssey
git cat-file -e "${ODYSSEY_COMMIT}^{commit}" 2>/dev/null || git fetch --quiet origin
git checkout --quiet --detach "${ODYSSEY_COMMIT}"
git --no-pager log -1 --format='%h %ad %s' --date=short
if [ ! -d third_party/nvdiffrast ]; then
  git clone --quiet --branch v0.4.0 --depth 1 https://github.com/NVlabs/nvdiffrast third_party/nvdiffrast
fi

# 2. Install the CUDA 12.8 toolkit, which the simulator's CUDA extensions are built with
if [ ! -x "${_runtime}/cuda-12.8/bin/nvcc" ]; then
  _installer=cuda_12.8.1_570.124.06_linux.run
  wget -q -P "${TMPDIR}" "https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/${_installer}"
  sh "${TMPDIR}/${_installer}" --silent --toolkit --toolkitpath="${_runtime}/cuda-12.8" --tmpdir="${TMPDIR}" \
    --no-man-page --override --no-drm --no-opengl-libs
  rm "${TMPDIR}/${_installer}"
fi

# 3. Install the simulator's environment with the package versions of Odyssey's reference host
_simulator_python="${_runtime}/envs/simulator/bin/python"
if [ ! -x "${_simulator_python}" ]; then
  "${_conda}" create -y -p "${_runtime}/envs/simulator" --override-channels -c conda-forge python=3.9.23 pip
  "${_simulator_python}" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  "${_simulator_python}" - >"${TMPDIR}/simulator_pins.txt" <<'PY'
import json
packages = json.load(open("OdysseyBenchmark/configs/deployment/environment-reference.json"))["simulator"]["packages"]
skipped = {"torch", "torchvision", "triton", "pip", "wheel", "nuplan-devkit", "nvdiffrast", "navsim", "mmcv"}
print("\n".join(f"{name}=={version}" for name, version in packages.items()
                if name.lower() not in skipped and not name.lower().startswith("nvidia-")))
PY
  "${_simulator_python}" -m pip install --no-deps -r "${TMPDIR}/simulator_pins.txt"
  "${_simulator_python}" -m pip install --no-deps \
    "git+https://github.com/motional/nuplan-devkit/@ce3c323af01c0d7ec5672f7832ef53f9c679aab0"
fi

# 4. Pick the compiler of the CUDA extensions: the system's, since the simulator loads the system's C++ runtime, which a
# newer compiler's build cannot link against, unless it is newer than CUDA 12.8 accepts (gcc 14; Ubuntu 26.04 has gcc
# 15). Then conda-forge's gcc 13, whose builds the newer system runtime still loads. odyssey.sh picks the same one.
_gxx="${_runtime}/gcc/bin/x86_64-conda-linux-gnu-g++"
if (( $(/usr/bin/gcc -dumpversion | cut -d. -f1) > 14 )) && [ ! -x "${_gxx}" ]; then
  "${_conda}" create -y -p "${_runtime}/gcc" --override-channels -c conda-forge gxx_linux-64=13
fi
if [ -x "${_gxx}" ]; then _cc="${_gxx%g++}gcc" _cxx="${_gxx}"; else _cc=/usr/bin/gcc _cxx=/usr/bin/g++; fi

# 5. Build nvdiffrast into it, for the GPU of this machine unless TORCH_CUDA_ARCH_LIST names others
if ! "${_simulator_python}" -m pip show nvdiffrast >/dev/null 2>&1; then
  PATH="$(dirname "${_simulator_python}"):${_runtime}/cuda-12.8/bin:${PATH}" CUDA_HOME="${_runtime}/cuda-12.8" \
    CC="${_cc}" CXX="${_cxx}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader)}" \
    "${_simulator_python}" -m pip install --no-deps --no-build-isolation ./third_party/nvdiffrast
fi

# 6. Install the environment of the Fixer, which restores the rendered camera images
_fixer_python="${_runtime}/fixer/bin/python"
if [ ! -x "${_fixer_python}" ]; then
  "${_conda}" create -y -p "${_runtime}/fixer" --override-channels -c conda-forge python=3.12 pip setuptools
  "${_fixer_python}" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  "${_fixer_python}" -m pip install --no-deps -r OdysseyRenderer/fixer/requirements_nodeps.txt
  "${_fixer_python}" -m pip install --no-deps --no-build-isolation OdysseyRenderer/fixer/flash_attn_stub
fi

# 7. Download the Fixer's weights
hf download ADRLAB/odyssey-models --include "fixer/*" --local-dir OdysseyRenderer
hf download nvidia/Fixer --include "base/*" --local-dir OdysseyRenderer/fixer/models
(cd OdysseyRenderer/fixer/models && sha256sum --quiet -c SHA256SUMS)
