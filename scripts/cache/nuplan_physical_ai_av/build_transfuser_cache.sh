#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-63

set -euo pipefail

# Point the data root to the default location
export PY123D_GARAGE_DATA_ROOT="${PY123D_GARAGE_DATA_ROOT:-data}"

# One thread per process: parallelism comes from the array shards and the loader workers, not from BLAS threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue

# Build this shard of the cache for every source; the mixture reads both caches
for _preset in tf_nuplan_trainval tf_physical_ai_av_trainval; do
	python -m py123d_garage.cache.build_cache \
		training@plan="py123d_garage.config.presets.python.training.transfuser:${_preset}" \
		hydra.run.dir="outputs/build_cache_transfuser_${_preset}/shard_${SLURM_ARRAY_TASK_ID:-0}" \
		shard_index="${SLURM_ARRAY_TASK_ID:-0}" \
		shard_count="${SLURM_ARRAY_TASK_COUNT:-1}" \
		cache_id="${SLURM_ARRAY_JOB_ID:-$(date +%s)}" \
		dataloader_config.num_workers="${SLURM_CPUS_PER_TASK:-8}" \
		"$@"
done
