#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=0 # the whole node's memory
#SBATCH --time=48:00:00

set -euo pipefail

# Important environment variables for py123d_garage for good training performance
export PY123D_GARAGE_DATA_ROOT="${PY123D_GARAGE_DATA_ROOT:-data}"
export PY123D_GARAGE_RUNTIME_TYPE_CHECKING=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Setup batch size per GPU
_global_batch_size=256
_num_nodes="${SLURM_JOB_NUM_NODES:-1}"
_gpus_per_node="${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}"

# On slurm, use srun to launch the training script, otherwise just use python
_launch=(python)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	_launch=(srun --kill-on-bad-exit=1 python)
fi

"${_launch[@]}" -m py123d_garage.training.train \
	training=py123d_garage.config.presets.python.training.transfuser:tf_physical_ai_av_trainvaltest \
	compile_mode=max-autotune \
	hydra.run.dir='outputs/pretrain_transfuser_physical_ai_av/${now:%Y.%m.%d.%H.%M.%S}' \
	policy_config.transfuser_config.planning_config.use_planning_decoder=false \
	lightning_trainer_config.devices="${_gpus_per_node}" \
	lightning_trainer_config.num_nodes="${_num_nodes}" \
	dataloader_config.batch_size="$((_global_batch_size / (_gpus_per_node * _num_nodes)))" \
	dataloader_config.num_workers="${SLURM_CPUS_PER_TASK:-16}" \
	log_images_every_n_steps=1000 \
	"$@"
