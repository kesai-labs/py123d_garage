#!/bin/bash

set -euo pipefail

HUGSIM_COMMIT="c18814c0ef2b1cfdeabe33506a5d345cfbb83cbc"
PIXI_VERSION="v0.81.0"

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
_runtime="${PWD}/lib/hugsim/runtime"
_tools="${PWD}/lib/hugsim/tools"
_conda="$(command -v micromamba || command -v conda)"
export TMPDIR="${_runtime}/tmp"
export PIXI_HOME="${_runtime}/pixi"
export PIXI_CACHE_DIR="${_runtime}/cache"
mkdir -p "${TMPDIR}"
# A conda env's activation adds its own -ccbin here, and nvcc refuses a second one.
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS

# 1. Clone valeo.ai's HUGSIM fork, which keeps the simulator's fixes and the benchmark runner; step 5 replaces its
# pixi.toml and pixi.lock, so they are restored first
if [ ! -d lib/hugsim/hugsim/.git ]; then
  git clone https://github.com/valeoai/HUGSIM_garage.git lib/hugsim/hugsim
fi
cd lib/hugsim/hugsim
git checkout --quiet -- pixi.toml pixi.lock
git cat-file -e "${HUGSIM_COMMIT}^{commit}" 2>/dev/null || git fetch --quiet origin
git checkout --quiet --detach "${HUGSIM_COMMIT}"
git --no-pager log -1 --format='%h %ad %s' --date=short

# 2. Install pixi, which installs the simulator's environment from the fork's lock file
if [ ! -x "${PIXI_HOME}/bin/pixi" ]; then
  curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION="${PIXI_VERSION}" PIXI_NO_PATH_UPDATE=1 bash
fi

# 3. Install the CUDA 12.8 toolkit, which the simulator's CUDA extensions are built with
if [ ! -x "${_runtime}/cuda-12.8/bin/nvcc" ]; then
  _installer=cuda_12.8.1_570.124.06_linux.run
  wget -q -P "${TMPDIR}" "https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/${_installer}"
  sh "${TMPDIR}/${_installer}" --silent --toolkit --toolkitpath="${_runtime}/cuda-12.8" --tmpdir="${TMPDIR}" \
    --no-man-page --override --no-drm --no-opengl-libs
  rm "${TMPDIR}/${_installer}"
fi

# 4. Pick the compiler of the CUDA extensions: the system's, unless it is newer than CUDA 12.8 accepts (gcc 14; Ubuntu
# 26.04 has gcc 15). Then conda-forge's gcc 13, whose builds the newer system C++ runtime still loads.
_gxx="${_runtime}/gcc/bin/x86_64-conda-linux-gnu-g++"
if (( $(/usr/bin/gcc -dumpversion | cut -d. -f1) > 14 )) && [ ! -x "${_gxx}" ]; then
  "${_conda}" create -y -p "${_runtime}/gcc" --override-channels -c conda-forge gxx_linux-64=13
fi
if [ -x "${_gxx}" ]; then _cc="${_gxx%g++}gcc" _cxx="${_gxx}"; else _cc=/usr/bin/gcc _cxx=/usr/bin/g++; fi

# 5. Install the simulator's environment from tools/pixi.toml and its lock file, building its CUDA extensions for the
# GPU of this machine unless TORCH_CUDA_ARCH_LIST names others. Without a lock file in tools yet, pixi resolves the
# manifest, and the lock file it writes is kept in tools: first without the CUDA extensions, whose metadata their
# setup.py builds with torch, which the environment then has.
_cuda_arch="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader)}"
# simple-knn uses FLT_MAX without including <cfloat>, which CUDA 12.8's headers no longer include for it.
_pixi_install() {
  PATH="${_runtime}/cuda-12.8/bin:${PATH}" CUDA_HOME="${_runtime}/cuda-12.8" CC="${_cc}" CXX="${_cxx}" \
    NVCC_APPEND_FLAGS="-include cfloat" \
    TORCH_CUDA_ARCH_LIST="${_cuda_arch}" TCNN_CUDA_ARCHITECTURES="${_cuda_arch//./}" \
    "${PIXI_HOME}/bin/pixi" install "$@"
}
if [ -f "${_tools}/pixi.lock" ]; then
  cp "${_tools}/pixi.toml" pixi.toml
  cp "${_tools}/pixi.lock" pixi.lock
  _pixi_install --frozen
else
  grep -v -E '^(simple-knn|gsplat|tinycudann|pytorch3d) = |^no-build-isolation = ' "${_tools}/pixi.toml" >pixi.toml
  _pixi_install
  cp "${_tools}/pixi.toml" pixi.toml
  _pixi_install
  cp pixi.lock "${_tools}/pixi.lock"
fi
