#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Odyssey drives the scenes under ODYSSEY_SCENES_ROOT on the nuPlan maps v1.1 under NUPLAN_MAPS_ROOT, the directory with nuplan-maps-v1.0.json.
: "${ODYSSEY_SCENES_ROOT:?set ODYSSEY_SCENES_ROOT to the download of ADRLAB/odyssey-scenes}"
: "${NUPLAN_MAPS_ROOT:?set NUPLAN_MAPS_ROOT to the nuPlan maps v1.1}"

# Declare input and output directories relative to py123d_garage; the checkpoint defaults to seed 0 of the newest posttrain run
_posttrain_run_dirs=(outputs/training/all/*/posttrain/)
_posttrain_checkpoint_files=("${_posttrain_run_dirs[-1]%/}"/model_0_*.pth)
_checkpoint_file="${CHECKPOINT_FILE:-${_posttrain_checkpoint_files[-1]}}"
[[ -f "${_checkpoint_file}" ]] || { echo "missing ${_checkpoint_file}" >&2; exit 1; }
_agent_config="$(realpath src/py123d_garage/evaluation/odyssey/configs/garage_transfuser.yaml)"
_out_dir="$(realpath -m "${OUTPUT_DIR:-outputs/evaluation/odyssey/$(date +%F_%H-%M-%S)}")"
# GPU indices as nvidia-smi lists them, and simulations at once on each of them.
_gpus="${GPUS:-0}"
_runs_per_gpu="${RUNS_PER_GPU:-1}"

# The agent config reads the checkpoint and the python that runs the policy from these.
CHECKPOINT_FILE="$(realpath "${_checkpoint_file}")"
PY123D_GARAGE_PYTHON="$(command -v python)"
export CHECKPOINT_FILE PY123D_GARAGE_PYTHON

# Odyssey's shell environment, over the checkout and environments of lib/odyssey/tools/setup_odyssey.sh.
export ODYSSEY_ROOT="${PWD}/lib/odyssey/odyssey"
export ODYSSEY_RUNTIME_ROOT="${PWD}/lib/odyssey/runtime"
export ODYSSEY_SIM_PY="${ODYSSEY_RUNTIME_ROOT}/envs/simulator/bin/python"
export ODYSSEY_FIXER_ROOT="${ODYSSEY_ROOT}/OdysseyRenderer/fixer"
export ODYSSEY_FIXER_PY="${ODYSSEY_RUNTIME_ROOT}/fixer/bin/python"
export CUDA_HOME="${ODYSSEY_RUNTIME_ROOT}/cuda-12.8"
export PATH="${ODYSSEY_RUNTIME_ROOT}/envs/simulator/bin:${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${ODYSSEY_ROOT}/OdysseyBenchmark:${ODYSSEY_ROOT}/OdysseyRenderer:${ODYSSEY_ROOT}/OdysseyTrafficAgent:${ODYSSEY_ROOT}/OdysseyBenchmark/odyssey_bridge:${ODYSSEY_ROOT}/third_party/nvdiffrast"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
TORCH_CUDA_ARCH_LIST="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader)"
export TORCH_CUDA_ARCH_LIST
# gsplat compiles its kernels here during the first episode, with the compiler setup_odyssey.sh picked: the system's,
# since the simulator loads the system's C++ runtime, which a newer compiler's build cannot link against, or the gcc 13
# it installed where the system's is newer than CUDA 12.8 accepts. A conda env's activation puts a second -ccbin into
# NVCC_PREPEND_FLAGS, which nvcc refuses.
export TORCH_EXTENSIONS_DIR="${ODYSSEY_RUNTIME_ROOT}/torch_extensions_${TORCH_CUDA_ARCH_LIST}"
_gxx="${ODYSSEY_RUNTIME_ROOT}/gcc/bin/x86_64-conda-linux-gnu-g++"
if [[ -x "${_gxx}" ]]; then
	export CC="${_gxx%g++}gcc" CXX="${_gxx}"
else
	export CC=/usr/bin/gcc CXX=/usr/bin/g++
fi
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
export TMPDIR="${ODYSSEY_RUNTIME_ROOT}/tmp"
# The Fixer's checkpoint is a full pickle.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# The simulator's numpy 1.23 bundles an OpenBLAS whose AVX-512 kernels return wrong pseudo-inverses on Sapphire Rapids
# CPUs; the controller's speed fit then commands accelerations of thousands of m/s^2.
export OPENBLAS_CORETYPE=Haswell
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
mkdir -p "${TMPDIR}"

# gsplat builds its kernels on its first import, and first deletes its build directory unless a finished build is in
# it, so simulations starting together delete each other's build. Build it here once, one job at a time.
mkdir -p "${TORCH_EXTENSIONS_DIR}"
flock "${TORCH_EXTENSIONS_DIR}/build.lock" "${ODYSSEY_SIM_PY}" -c "import gsplat.cuda._backend"

# Every simulation running at once needs its own copy of the maps.
_map_slots="$(bash "${ODYSSEY_ROOT}/OdysseyBenchmark/scripts/make_map_slots.sh" \
	"${NUPLAN_MAPS_ROOT}" "${ODYSSEY_RUNTIME_ROOT}/maps" "${_gpus}" "${_runs_per_gpu}")"

# Run the scenes (debug: three, mini: 30, all: the benchmark's 100) in both traffic modes, then print the results sheet.
cd "${ODYSSEY_ROOT}"
"${ODYSSEY_SIM_PY}" -m odyssey_runtime eval \
	--agent "${_agent_config}" \
	--gpus "${_gpus}" \
	--runs-per-gpu "${_runs_per_gpu}" \
	--map-slots "${_map_slots}" \
	--scenes "${SCENES:-debug}" \
	--output "${_out_dir}" \
	"$@"
"${ODYSSEY_SIM_PY}" OdysseyBenchmark/tools/merge_results.py -f "${_out_dir}"
