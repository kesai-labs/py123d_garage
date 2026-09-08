<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/logo_dark.png">
  <img src="../assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">Data Overview</h1>

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
└── physical-ai-av/
    └── 123D/
        └── logs/
            ├── physical-ai-av_train/
            │   └── 0000b6ee-e657-4369-9de3-a53031ea4ba9/
            ├── physical-ai-av_val/
            └── physical-ai-av_test/
```

Feature caches are written next to the logs under `$PY123D_GARAGE_DATA_ROOT/<dataset>/py123d_garage_cache/<cache name>`.

## Access data

To download or convert each dataset into this layout, see:

- [nuPlan](nuplan.md)
- [Physical AI AV](physical_ai_av.md)
- [CARLA](carla.md)
