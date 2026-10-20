#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-31

set -euo pipefail

# Point the data root to the default location
export PY123D_GARAGE_DATA_ROOT="${PY123D_GARAGE_DATA_ROOT:-data}"

# One thread per process: parallelism comes from the array shards and the loader workers, not from BLAS threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue

# Build this shard of the cache, shared by all three stages
python -m py123d_garage.cache.build_cache \
	training@plan=py123d_garage.config.presets.training.latent_transfuser:pretrain \
	hydra.run.dir="outputs/cache/shard_${SLURM_ARRAY_TASK_ID:-0}" \
	shard_index="${SLURM_ARRAY_TASK_ID:-0}" \
	shard_count="${SLURM_ARRAY_TASK_COUNT:-1}" \
	cache_id="${SLURM_ARRAY_JOB_ID:-$(date +%s)}" \
	dataloader_config.num_workers="${SLURM_CPUS_PER_TASK:-8}" \
	"$@"
