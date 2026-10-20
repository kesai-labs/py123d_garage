<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">Data Overview</h1>

## Datasets at a glance

| Dataset                             | Camera hours | Sensors                 | Labels                                                 | Maps      | Used for training                 |
| :---------------------------------- | -----------: | :---------------------- | :----------------------------------------------------- | :-------- | :-------------------------------- |
| [nuPlan](nuplan.md)                 |         ~135 | 8 cameras, lidar        | 3D boxes, traffic lights                               | 4 cities  | train, val                        |
| [Physical AI AV](physical_ai_av.md) |       ~1,650 | 7 cameras, lidar        | machine-labeled 3D boxes                               | none      | all splits, minus the NuRec clips |
| [NuRec](nurec.md)                   |           ~9 | 7 cameras, lidar        | 3D boxes from reconstruction                           | per scene | train, other                      |
| [CARLA](carla.md)                   | ~85 per view | 6 cameras, lidar, radar | 3D boxes, traffic lights, depth, semantic segmentation | 12 towns  | normal view                       |

## Expected layout

All datasets are read in the [123D](https://github.com/kesai-labs/py123d) format. Point `PY123D_GARAGE_DATA_ROOT` at a directory with one dataset per subfolder:

```
$PY123D_GARAGE_DATA_ROOT/
├── nuplan/
│   └── 123D/
│       ├── logs/
│       │   ├── nuplan_train/
│       │   │   └── 2021.05.12.19.36.12_veh-35_00005_00204/
│       │   ├── nuplan_val/
│       │   └── nuplan_test/
│       └── maps/
│           └── nuplan/
│               ├── nuplan_us-nv-las-vegas-strip.arrow
│               └── ...
├── lead/
│   └── 123D/
│       ├── logs/
│       │   └── normal_view/
│       │       ├── Accident/
│       │       │   └── Town03_Rep0_route_001783_route0_08_02_07_21_33/
│       │       └── ...
│       └── maps/
│           └── carla/
│               ├── carla_town03.arrow
│               └── ...
├── physical-ai-av/
│   └── 123D/
│       └── logs/
│           ├── physical-ai-av_train/
│           │   └── 0000b6ee-e657-4369-9de3-a53031ea4ba9/
│           ├── physical-ai-av_val/
│           └── physical-ai-av_test/
└── nurec/
    └── 123D/
        └── logs/
            ├── nurec_train/
            │   └── 00040136-e651-4abd-991d-0655ccda9430/
            ├── nurec_val/
            └── nurec_other/
```

The training recipe reads Physical-AI-AV, NuRec and nuPlan. Training never sees the evaluation scenes: it skips `nurec_val` and `nuplan_test`, as well as every Physical-AI-AV clip that NuRec reconstructs.

The training cache is written next to the logs, under `$PY123D_GARAGE_DATA_ROOT/<dataset>/py123d_garage_cache/`.

## Access data

To download or convert each dataset into this layout, see:

- [nuPlan](nuplan.md)
- [Physical AI AV](physical_ai_av.md)
- [NuRec](nurec.md)
- [CARLA](carla.md)
