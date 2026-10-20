<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">Physical AI AV Dataset</h1>

### Overview

Physical AI AV holds about 1,700 hours of driving from 25 countries and more than 2,500 cities, cut into 20 s clips. Each clip has seven cameras and a 360° lidar, both at 10 Hz in the 123D copy; the training recipe uses the front wide camera. The labels are machine-generated 3D boxes and egomotion, which covers about 140 s around each clip. There are no maps. The license does not allow redistribution, so no sample images are shown here; see the [dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).

| Split                  |   Clips | Camera hours | Used for training |
| :--------------------- | ------: | -----------: | :---------------- |
| `physical-ai-av_train` | 149,774 |         ~830 | Yes               |
| `physical-ai-av_val`   |  88,513 |         ~490 | Yes               |
| `physical-ai-av_test`  |  60,039 |         ~330 | Yes               |

Training uses all three splits, since the closed-loop evaluation runs on NuRec scenes, not on these splits. It skips every clip that NuRec reconstructs: those clips are read through [NuRec](nurec.md), whose validation scenes must stay unseen.

The NVIDIA AV Dataset License does not allow redistribution, so no 123D version is hosted; you need to convert the raw data yourself.

### Convert to 123D yourself

Download the raw data from [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) and convert it. See documentation of [py123d](https://kesai.eu/py123d/datasets/physical-ai-av/#conversion).

```bash
py123d-conversion datasets=["physical-ai-av"] dataset_paths.physical_ai_av_data_root=<raw download>
```

Write the output to `$PY123D_GARAGE_DATA_ROOT/physical-ai-av/123D`.

Our copy converts all three splits and all seven cameras, straight from the Hub's `chunk_*.zip` archives. Clips without lidar or egomotion, about 4 to 5 percent, are skipped.

### Expected layout

There are no maps. Log names are the clip UUIDs.

```
$PY123D_GARAGE_DATA_ROOT/physical-ai-av/123D/
└── logs/
    ├── physical-ai-av_train/
    │   └── 0000b6ee-e657-4369-9de3-a53031ea4ba9/
    │       ├── box_detections_se3.arrow
    │       ├── camera.ftcam_f0.arrow
    │       ├── camera.ftcam_l0.arrow
    │       ├── camera.ftcam_l1.arrow
    │       ├── camera.ftcam_r0.arrow
    │       ├── camera.ftcam_r1.arrow
    │       ├── camera.ftcam_tele_b0.arrow
    │       ├── camera.ftcam_tele_f0.arrow
    │       ├── ego_state_se3.arrow
    │       ├── lidar.lidar_merged.arrow
    │       ├── route_position.arrow
    │       └── sync.arrow
    ├── physical-ai-av_val/
    └── physical-ai-av_test/
```
