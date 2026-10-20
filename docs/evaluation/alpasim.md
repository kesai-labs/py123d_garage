<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">AlpaSim evaluation</h1>

AlpaSim is NVIDIA's closed-loop simulator and the main benchmark of this repository. It splits a simulation into services: sensor rendering, traffic, scoring, and the driver. py123d_garage provides only the driver, a small server around a checkpoint. The AlpaSim wizard starts the other services in Docker, connects them to the driver and writes the scores.

AlpaSim is its own Python project with its own environment. So we set up AlpaSim and install py123d_garage into it, not the other way round.

## Setup

You need Docker with the NVIDIA runtime, `uv`, and a Hugging Face token with access to [PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) for the scenes.

```bash
lib/alpasim/tools/setup_alpasim.sh
export HF_TOKEN=<token>
```

The script clones the `e2e_challenge` branch of [NVlabs/alpasim](https://github.com/NVlabs/alpasim/tree/e2e_challenge) at a pinned commit into `lib/alpasim/alpasim` and installs py123d_garage into its environment. Read [its documentation](https://github.com/NVlabs/alpasim/tree/e2e_challenge/docs) to learn the wizard.

## Simulator patches

The setup script also patches AlpaSim, with the patches in `lib/alpasim/tools/patches`. The patches only render less, to save compute. The simulation and the frames the policy sees stay the same as on the challenge server.

## Local evaluation

The challenge has two tracks. The Physical-AI-AV track drives NuRec scenes; the nuPlan track drives MTGS scenes.

```bash
scripts/evaluation/alpasim_pai.sh                                    # Physical-AI-AV track
ALPASIM_NUPLAN_ROOT=<mtgs root> scripts/evaluation/alpasim_nuplan.sh   # nuPlan track
```

- By default the script drives seed 0 of the newest posttrain run, with the camera rig of the matching training source. Set `CHECKPOINT_FILE` to drive another checkpoint.
- It drives one scene. Set `SCENE_IDS` for others.
- On the nuPlan track, `CAMERAS=nuplan_1cam` renders only the front camera, which is all the released policies read.
- The Physical-AI-AV track renders without the NuRec harmonizer, like the challenge. The README's numbers render with it, as AlpaSim's Alpamayo 1 reference run does: `PRESET="deploy=local topology=1gpu +garage_renderer=nre_harmonized" scripts/evaluation/alpasim_pai.sh`.
- The scores are written to `outputs/evaluation/alpasim/<track>/<timestamp>/aggregate/results-summary.json`.
- The nuPlan track needs the MTGS assets from [OpenDriveLab/AlpasimChallenge2026_nuplan_track](https://huggingface.co/datasets/OpenDriveLab/AlpasimChallenge2026_nuplan_track), extracted under `ALPASIM_NUPLAN_ROOT`.

## Scene splits

The Physical-AI-AV track uses AlpaSim's curated NuRec suites, listed in [sim_suites_curated.csv](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/data/scenes/sim_suites_curated.csv): `nurec_curated_train` with 1,761 scenes and `nurec_curated_val` with 441 scenes. The 441 validation scenes are the evaluation set; training never sees them (see [NuRec](../data/nurec.md)).

## The route

AlpaSim builds one route per scene from the recorded drive. It does not change with how the policy drives. Each step, the simulator sends only the section from 40 m to 80 m ahead of the ego, so a policy cannot read the route closer than 40 m. The released policy reads target points at 40, 60 and 80 m. It is trained with them mostly dropped, so it can also drive without a route.

The driver plans at 2 Hz and serves the simulator's 10 Hz calls from the stored plan.

## Driver options

Pass driver settings in `DRIVER_OVERRIDES`, for example `DRIVER_OVERRIDES="route_memory=true"`. The same variable works for the submission image.

- `route_memory=true`: remember the route of earlier steps, so the section closer than 40 m is known once the ego has driven past it.
- `extrapolate_route=true`: extend a route that ends before the farthest target point straight along its last heading.
- `ego_hood=<name>`: paste the hood of a training vehicle onto the front camera, so the rendered frames look like the training data. Names are the AlpaSim rigs `hyperion_8` and `hyperion_8_1`, and the Physical-AI-AV vehicles `pai_00` to `pai_14`.

## Challenge submission

Register at the [challenge space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026), then run the steps in order:

```bash
scripts/alpasim_submission/login_challenge.sh
scripts/alpasim_submission/build_docker_image.sh
scripts/alpasim_submission/smoke_test_image.sh
VERSION=v1 scripts/alpasim_submission/submit_image.sh
```

- The image holds seed 0 of the newest posttrain run by default. `CHECKPOINT_FILES` with several space-separated checkpoints of one run bakes an ensemble, and the driver averages their plans.
- `TRACK=nuplan` builds, tests and submits for the nuPlan track. The default is `pai`. Each track needs its own image, because the image holds the camera rig of its track.
- The challenge rejects a reused tag, so give each submission a new `VERSION`.
