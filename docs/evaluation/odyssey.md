<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">Odyssey evaluation</h1>

[Odyssey](https://github.com/SNU-ADR/Odyssey) is a closed-loop benchmark of Seoul National University on 100 nuPlan logs of 100 seconds each, reconstructed with Gaussian splatting and restored with NVIDIA's Fixer. Each scene runs twice, once with the logged traffic and once with reactive traffic, and the main score is RouteDS ([paper](https://arxiv.org/abs/2610.06469)). Its scores are not comparable to the nuPlan track of AlpaSim.

Odyssey runs its simulator, the image restorer and the policy as separate processes, each in its own environment. py123d_garage provides the policy's process: an adapter that drives a checkpoint in the py123d_garage environment.

## Setup

You need micromamba or conda, an NVIDIA GPU, the nuPlan maps v1.1, and a Hugging Face account that accepted the terms of [ADRLAB/odyssey-scenes](https://huggingface.co/datasets/ADRLAB/odyssey-scenes).

```bash
lib/odyssey/tools/setup_odyssey.sh
hf download ADRLAB/odyssey-scenes --repo-type dataset --local-dir <scenes dir>
```

The script clones [SNU-ADR/Odyssey](https://github.com/SNU-ADR/Odyssey) at a pinned commit into `lib/odyssey/odyssey` and installs the CUDA 12.8 toolkit, the simulator and Fixer environments and the Fixer weights into `lib/odyssey/runtime`, about 25 GB. The scenes take another 217 GB.

## Evaluation

```bash
ODYSSEY_SCENES_ROOT=<scenes dir> NUPLAN_MAPS_ROOT=<nuplan maps dir> scripts/evaluation/odyssey.sh
```

- By default the script drives seed 0 of the newest posttrain run. Set `CHECKPOINT_FILE` to drive another checkpoint.
- It drives three scenes. `SCENES=mini` drives 30, `SCENES=all` the benchmark's 100.
- `GPUS=0,1,2,3` spreads the scenes over GPUs, and `RUNS_PER_GPU=2` runs two simulations on each. One simulation takes 15 to 25 GB of GPU memory.
- The full benchmark takes about 50 GPU-hours. A rerun with the same `OUTPUT_DIR` continues an interrupted one.
- The results sheet is printed at the end and written to `outputs/evaluation/odyssey/<timestamp>/summary.csv`.

## The policy in Odyssey

Odyssey serves a route from a standard-definition map: a line along the road, not along a lane. The policy reads its target points at 40, 60 and 80 m along that line. It plans every 0.5 s, and the simulator's 10 Hz steps follow the stored plan in between.

The agent config `src/py123d_garage/evaluation/odyssey/configs/garage_transfuser.yaml` declares what the released policies read: the front camera now and 1 s ago. A checkpoint that reads other cameras needs its own agent config.
