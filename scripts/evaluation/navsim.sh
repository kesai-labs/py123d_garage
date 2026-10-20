#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

set -euo pipefail

# Point the data root to the default location
export PY123D_GARAGE_DATA_ROOT="${PY123D_GARAGE_DATA_ROOT:-data}"

# One thread per process: parallelism comes from the worker processes, not from BLAS threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue

# Declare input files relative to py123d_garage; the checkpoint defaults to seed 0 of the newest posttrain run
_posttrain_run_dirs=(outputs/training/all/*/posttrain/)
_posttrain_checkpoint_files=("${_posttrain_run_dirs[-1]%/}"/model_0_*.pth)
_checkpoint_file="${CHECKPOINT_FILE:-${_posttrain_checkpoint_files[-1]}}"
[[ -f "${_checkpoint_file}" ]] || { echo "missing ${_checkpoint_file}" >&2; exit 1; }

# Run the evaluation on the navtest split
python -m py123d_garage.evaluation.navsim.evaluate \
	hydra.run.dir='outputs/evaluation/navsim/${now:%Y.%m.%d.%H.%M.%S}' \
	policy_config.evaluation_checkpoint_files="[${_checkpoint_file}]" \
	parallelization_config.device=cuda \
	parallelization_config.inference_batch_size=8 \
	parallelization_config.max_workers=8 \
	"$@"
