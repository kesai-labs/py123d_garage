<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">NavSafe-∞ evaluation</h1>

[NavSafe-∞](https://github.com/VAIL-UCLA/NavSafe-infty) is a closed-loop safety benchmark of 280 scenarios on reconstructed nuPlan logs, 10 for each of 28 safety event types in four categories: traffic crashes, crashes with vulnerable road users, traffic violations and traffic incidents. 66 scenarios insert actors such as animals, cyclists or debris into the scene. Isaac Sim simulates the ego and the traffic, and NVIDIA's NuRec renderer draws the cameras ([paper](https://arxiv.org/abs/2609.26618), [leaderboard](https://navsafe-vail.github.io/#leaderboard)).

NavSafe calls the policy through an adapter in its own environment. py123d_garage installs one there, which starts an agent process that drives a checkpoint in the py123d_garage environment.

## Setup

You need NVIDIA GPUs with RT cores (Isaac Sim does not run on A100 or H100), about 2.2 TB of disk for the scenarios, and the NuRec renderer image `nvcr.io/nvidia/nre/nre-ga:26.04`: with Docker and an [NGC API key](https://org.ngc.nvidia.com/setup/api-key), or as an enroot `.sqsh` file on Slurm clusters with pyxis.

```bash
lib/navsafe/tools/setup_navsafe.sh
hf download c13752hz/NavSafe --repo-type dataset --local-dir <data dir> \
  --include "full_test/*" --include "asset/*" --include "gait_bank/*"
```

The script clones NavSafe at a pinned commit into `lib/navsafe/navsafe`, installs its locked environment with Isaac Sim and Isaac Lab, and our adapter into it.

## Evaluation

```bash
NAVSAFE_DATA_ROOT=<data dir> NGC_ENV_FILE=<file that sets NGC_API_KEY> scripts/evaluation/navsafe.sh
```

- By default the script drives seed 0 of the newest posttrain run. Set `CHECKPOINT_FILE` to drive another checkpoint.
- It drives all 280 scenarios with NavSafe's protocol: semi-reactive traffic, LQR tracking, replanning at 2 Hz, and the episode ends at the first contact.
- `GPUS=0,1,2,3` splits the scenarios over lanes, one per GPU, each with its own renderer.
- Without Docker, `RENDERER_SQSH=<nre-ga_26.04.sqsh>` runs one renderer per lane as a pyxis step of the Slurm job instead of a container per scenario.
- `NAVSAFE_TOKENS=<file>` drives only the scenarios listed in it, and `EVAL_SEED` sets the policy's seed.
- A rerun with the same `OUTPUT_DIR` continues an interrupted one.
- The scores are written to `outputs/evaluation/navsafe/<timestamp>/report.md`.

## The policy in NavSafe

NavSafe serves a left, straight or right command, but no route, so the released policies drive without target points. The policy plans every 0.5 s and NavSafe tracks the plan 4 s ahead.

NavSafe renders nuPlan's front camera of the reconstructed logs, 40 rows taller than nuPlan's. The adapter cuts it to nuPlan's 1920 x 1080 frame, and the policy reads it as nuPlan's front camera, now and 1 s ago.
