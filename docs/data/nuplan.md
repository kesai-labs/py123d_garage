<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/logo_dark.png">
  <img src="../assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">nuPlan Dataset</h1>

### Download data in 123D format

nuPlan v1.1 is hosted in 123D format on Hugging Face as [kesai-labs/nuplan](https://huggingface.co/datasets/kesai-labs/nuplan). The baseline policies read the sensor logs, their three front cameras, the lidar, the metadata and the maps:

```bash
hf download kesai-labs/nuplan --repo-type dataset --local-dir $PY123D_GARAGE_DATA_ROOT/nuplan/123D \
    --include 'logs/*' 'maps/*' \
    --exclude '*/camera.pcam_b0.arrow' '*/camera.pcam_l1.arrow' '*/camera.pcam_l2.arrow' \
              '*/camera.pcam_r1.arrow' '*/camera.pcam_r2.arrow'
```

Drop `--exclude` to also fetch the five other cameras. The sensorless logs under `logs_sensorless/` carry no camera or lidar, so no policy here can train on them.

### Convert to 123D yourself

Alternatively, download the raw data from [nuplan.org](https://www.nuplan.org/download) and convert it yourself. See documentation of [py123d](https://kesai.eu/py123d/datasets/nuplan/#conversion).

The release was converted with py123d's `nuplan` parser from the v1.1 DBs and sensor archives, synced at 10 Hz on the box detection timestamps, which equal the lidar timestamps. Two settings differ from the converter's defaults, which store sensors as paths into the raw tree: cameras are embedded as JPEG at OpenCV quality 75 (`camera_store_option: jpeg_binary`), and lidar sweeps as laz with 2 cm xyz quantization (`lidar_store_option: binary`, `lidar_codec: laz`, `laz_scales: [0.02, 0.02, 0.02]`). `route_position` is derived from the ego odometry after conversion and needs py123d >= 0.7.0.

### Expected layout

```
$PY123D_GARAGE_DATA_ROOT/nuplan/123D/
├── logs/
│   ├── nuplan_train/
│   │   └── 2021.05.12.19.36.12_veh-35_00005_00204/
│   │       ├── box_detections_se3.arrow
│   │       ├── camera.pcam_f0.arrow
│   │       ├── camera.pcam_l0.arrow
│   │       ├── camera.pcam_r0.arrow
│   │       ├── custom.scenario.arrow
│   │       ├── ego_state_se3.arrow
│   │       ├── lidar.lidar_merged.arrow
│   │       ├── route_position.arrow
│   │       ├── sync.arrow
│   │       └── traffic_light_detections.arrow
│   ├── nuplan_val/
│   └── nuplan_test/
└── maps/
    └── nuplan/
        ├── nuplan_sg-one-north.arrow
        ├── nuplan_us-ma-boston.arrow
        ├── nuplan_us-nv-las-vegas-strip.arrow
        └── nuplan_us-pa-pittsburgh-hazelwood.arrow
```
