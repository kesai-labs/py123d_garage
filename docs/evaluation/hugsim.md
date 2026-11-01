<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">HUGSIM evaluation</h1>

[HUGSIM](https://github.com/hyzhou404/HUGSIM) is a closed-loop simulator on 82 scenes of nuScenes, PandaSet, KITTI-360 and Waymo, reconstructed with Gaussian splatting, with reconstructed vehicles as the other traffic. Its benchmark has 436 scenarios in four levels of difficulty, and the main score is the HD-Score ([paper](https://arxiv.org/abs/2412.01718)). py123d_garage runs it through [valeoai/HUGSIM_garage](https://github.com/valeoai/HUGSIM_garage), a maintained fork with fixes to the simulator and its scorer, a benchmark runner and a second score of the driven path.

HUGSIM runs its simulator and the policy as separate processes, each in its own environment. py123d_garage provides the policy's process: an agent that drives a checkpoint in the py123d_garage environment.

## Setup

You need an NVIDIA GPU and the raw [nuScenes](https://www.nuscenes.org), whose HD maps 16 of the nuScenes scenarios read.

```bash
lib/hugsim/tools/setup_hugsim.sh
lib/hugsim/hugsim/.pixi/envs/default/bin/python lib/hugsim/hugsim/prepare_benchmark.py --root <data dir> --from-hf
```

The script clones HUGSIM_garage at a pinned commit into `lib/hugsim/hugsim`, and installs pixi, the CUDA 12.8 toolkit and the simulator's environment. `prepare_benchmark.py` downloads the vehicles, scenes and scenarios, about 41 GB.

## Evaluation

```bash
HUGSIM_DATA=<data dir> NUSCENES_RAW=<nuscenes dir> scripts/evaluation/hugsim.sh
```

- By default the script drives seed 0 of the newest posttrain run. Set `CHECKPOINT_FILE` to drive another checkpoint.
- It drives the first scenario of each dataset. `LIMIT=0` drives all 436, `LIMIT=10` the first 10 of each dataset.
- `SET=prechallenge345` drives the 345 scenarios of the pre-challenge release instead, the set older papers report.
- `GPUS=0,1,2,3` spreads the scenarios over GPUs, and `RUNS_PER_GPU=4` runs four episodes on each.
- A rerun with the same `OUTPUT_DIR` continues an interrupted one.
- The scores are printed at the end and written to `outputs/evaluation/hugsim/<timestamp>/summary.json`.

## The policy in HUGSIM

HUGSIM serves no route, only a left, straight or right command, and the released policies read no command. They drive without target points, as most published HUGSIM baselines do: those get the straight command at every step. The policy plans every 0.5 s, and the simulator's 0.25 s steps follow the stored plan in between. HUGSIM scores the plan 4 s ahead.

HUGSIM renders one camera rig on every dataset. Its front camera has about the field of view of nuPlan's, and the policy reads it as nuPlan's front camera, now and 1 s ago.

Most published HUGSIM numbers predate HUGSIM_garage's fixes, which change the scores. To compare with them, pass `--before_fixes` to the script, which hands it to the simulator.
