<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/logo_dark.png">
  <img src="../assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">Physical AI AV Dataset</h1>

The NVIDIA AV Dataset License does not allow redistribution, so no 123D version is hosted; you need convert the raw data yourself.

### Convert to 123D yourself

Download the raw data from [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) and convert it. See documentation of [py123d](https://kesai.eu/py123d/datasets/physical-ai-av/#conversion).

```bash
py123d-conversion datasets=["physical-ai-av"] dataset_paths.physical_ai_av_data_root=<raw download>
```

Write the output to `$PY123D_GARAGE_DATA_ROOT/physical-ai-av/123D`.

Our copy was converted with py123d's `physical-ai-av` parser straight from the Hub's `chunk_*.zip` archives, without extracting them: all three splits, all seven cameras, synced at 10 Hz on the lidar spin grid, ego and box dynamics inferred. Cameras are embedded as JPEG at OpenCV quality 75, lidar spins as laz with 2 cm xyz quantization (`laz_scales: [0.02, 0.02, 0.02]`, the default is 1 cm). Clips that ship without lidar or egomotion, about 4 to 5 percent, are skipped.

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
