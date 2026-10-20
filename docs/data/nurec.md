<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">NuRec Dataset</h1>

NuRec holds the Physical-AI-AV clips that NVIDIA reconstructed for AlpaSim. The AlpaSim evaluation renders its scenes from these reconstructions. Training uses the recorded data of the same clips: the real camera frames and lidar, plus the map and boxes of each reconstruction.

| Split         | Clips | Camera hours | Used for training                                 |
| :------------ | ----: | -----------: | :------------------------------------------------ |
| `nurec_train` | 1,171 |         ~6.5 | Yes                                               |
| `nurec_other` |   129 |         ~0.7 | Yes                                               |
| `nurec_val`   |   294 |         ~1.6 | No: held out for the AlpaSim Physical-AI-AV track |

`nurec_val` covers 294 of the 441 scenes in AlpaSim's curated validation suite. Of the rest, 137 have no clip in the Physical-AI-AV release, 9 have no usable map and 1 failed to convert. As with Physical AI AV, the license does not allow redistribution, so no sample images are shown.

### Download and convert

The dataset is gated. Accept its license on [nvidia/PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec), then download and convert it with py123d. See the documentation of [py123d](https://kesai.eu/py123d/datasets/nurec/). The camera frames and lidar come from the [Physical-AI-AV](physical_ai_av.md) download of the same clips.

Write the output to `$PY123D_GARAGE_DATA_ROOT/nurec/123D`.

### Splits

- `nurec_train` and `nurec_val` follow AlpaSim's curated scene suites, [sim_suites_curated.csv](https://github.com/NVlabs/alpasim/blob/85d5cb13f39a24bc73621861a2c09b3fddf5ce27/data/scenes/sim_suites_curated.csv). `nurec_other` holds the remaining scenes.
- Training reads `nurec_train` and `nurec_other`. `nurec_val` is held out for evaluation.

### Expected layout

Each log carries its own map. Log names are the clip UUIDs.

```
$PY123D_GARAGE_DATA_ROOT/nurec/123D/
└── logs/
    ├── nurec_train/
    │   └── 00040136-e651-4abd-991d-0655ccda9430/
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
    │       ├── map.arrow
    │       ├── route_position.arrow
    │       └── sync.arrow
    ├── nurec_val/
    └── nurec_other/
```
