<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/logo_dark.png">
  <img src="../assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">Alpasim evaluation</h1>

Alpasim is NVIDIA's closed-loop simulator. It comes as its own Python project with its own environment, so evaluating there works the other way round from the rest of this repository: we set up the Alpasim environment and install py123d_garage into it as a dependency.

At evaluation time the garage plays only one role, the driver. Alpasim splits a simulation into services: rendering the sensors, simulating the traffic, scoring, and the driver, which receives the sensor data and returns the trajectory to follow. The garage implements that driver as a small server around a checkpoint. Everything else is handled by the Alpasim wizard, which starts the remaining services in Docker, connects them to our driver, and writes the scores.

## Setup

Needs Docker with the NVIDIA runtime, `uv`, and a Hugging Face token with access to [PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) for the scene downloads.

```bash
lib/alpasim/tools/setup_alpasim.sh
export HF_TOKEN=<token>
```

The script clones the `e2e_challenge` branch of [NVlabs/alpasim](https://github.com/NVlabs/alpasim/tree/e2e_challenge) at a pinned commit into `lib/alpasim/alpasim` (gitignored), syncs its venv and installs `py123d_garage` into its venv. Every Alpasim command below runs from that venv.

## Important files

| Path                                               | What it is                                                                                                                                             |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lib/alpasim/tools/setup_alpasim.sh`               | Sets up the Alpasim checkout, see above                                                                                                                |
| `lib/alpasim/tools/submission.Dockerfile`          | Image for the challenge submission                                                                                                                     |
| `src/py123d_garage/evaluation/alpasim/evaluate.py` | The driver                                                                                                                                             |
| `src/py123d_garage/evaluation/alpasim/configs/`    | Hydra configs that tell the wizard about our driver: which cameras it wants (`driver`), where it listens (`driver_source`), the nuPlan rig (`cameras`) |

## Local evaluation

To get familiar with Alpasim, read [its documentation](https://github.com/NVlabs/alpasim/tree/e2e_challenge/docs) first. Py123D Garage provides only the driver.

### Physical AI AV Track

We provide an small script to start the py123d_garage's driver and evaluate small scenes locally.

```bash
CHECKPOINT_FILE=<checkpoint.pth> scripts/evaluation/alpasim/latent_transfuser.sh
```

### nuPlan track

The nuPlan track renders with MTGS. Download the trajdata cache, the scene configs and the asset shards. Then drive:

```bash
ALPASIM_NUPLAN_ROOT=<mtgs root> CHECKPOINT_FILE=<checkpoint.pth> scripts/evaluation/alpasim/latent_transfuser_nuplan.sh
```

## Challenge submission

Register at the [challenge space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026), then run the steps in order:

```bash
scripts/alpasim_submission/login_challenge.sh
scripts/alpasim_submission/build_docker_image.sh
scripts/alpasim_submission/smoke_test_image.sh
scripts/alpasim_submission/submit_image.sh
```
