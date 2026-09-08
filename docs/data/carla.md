<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/logo_dark.png">
  <img src="../assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">CARLA Dataset</h1>

### Download data

CARLA logs are collected with [LEAD](https://github.com/kesai-labs/lead) and hosted in 123D format on Hugging Face as [ln2697/lead-123d](https://huggingface.co/datasets/ln2697/lead-123d). The normal view and the maps are enough for training and evaluation:

```bash
hf download ln2697/lead-123d --repo-type dataset --local-dir $PY123D_GARAGE_DATA_ROOT/lead/123D \
    --include 'logs/normal_view/*' 'maps/*' 'config.yaml'
```

Drop `--include` to also fetch the perturbed camera views.

### Collect yourself

There is no conversion step: LEAD's privileged expert drives the Leaderboard 2.0 routes and writes 123D directly, at the 20 Hz tick rate for point clouds and scene state and every fifth tick for the cameras. See the LEAD repository for the data collection scripts.

### Expected layout

```
$PY123D_GARAGE_DATA_ROOT/lead/123D/
├── config.yaml
├── logs/
│   └── normal_view/
│       ├── Accident/
│       │   └── Town03_Rep0_route_001783_route0_08_02_07_21_33/
│       │       ├── box_detections_se3.arrow
│       │       ├── camera.pcam_{f0,b0,l0,l1,r0,r1}.arrow
│       │       ├── camera_depth.pcam_{f0,b0,l0,l1,r0,r1}.arrow
│       │       ├── camera_instance.pcam_{f0,b0,l0,l1,r0,r1}.arrow
│       │       ├── camera_semantic.pcam_{f0,b0,l0,l1,r0,r1}.arrow
│       │       ├── custom.driving_meta.arrow
│       │       ├── ego_state_se3.arrow
│       │       ├── lidar.lidar_top.arrow
│       │       ├── radar.radar_merged.arrow
│       │       ├── sync.arrow
│       │       └── traffic_light_detections.arrow
│       └── ... (43 scenario types)
└── maps/
    └── carla/
        ├── carla_town01.arrow
        └── ... (Town01 through Town15)
```
