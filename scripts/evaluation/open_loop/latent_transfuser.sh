#!/usr/bin/env bash

set -euo pipefail

# Point the data root to the default location
export PY123D_GARAGE_DATA_ROOT="${PY123D_GARAGE_DATA_ROOT:-data}"

# One thread per process: parallelism comes from the worker processes, not from BLAS threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue

# Declare input files relative to py123d_garage
_checkpoint_file="${CHECKPOINT_FILE:-outputs/checkpoints/nuplan_latent_transfuser/model.pth}"

# Run the evaluation on the logged trajectories
python -m py123d_garage.evaluation.open_loop.evaluate \
	hydra.run.dir='outputs/evaluation/open_loop/latent_transfuser/${now:%Y.%m.%d.%H.%M.%S}' \
	policy_config.evaluation_checkpoint_file="${_checkpoint_file}" \
	parallelization_config.device=cuda \
	parallelization_config.inference_batch_size=8 \
	parallelization_config.max_workers=8 \
	"$@"
