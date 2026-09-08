<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/logo_dark.png">
  <img src="../assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">NAVSIM evaluation</h1>

> \[!CAUTION\]
> This is not the official NAVSIM but a reimplementation. The PDM score is rewritten on 123D, and the policy is conditioned on target points instead of the driving command. Scores are not comparable to the NAVSIM leaderboard.

NAVSIM scores a checkpoint with the NAVSIM PDM score on the navtest split. The predicted trajectory is simulated for 4 seconds and scored on collisions, drivable area, driving direction, time to collision, progress and comfort.

## Usage

To use this benchmark, you needs the nuPlan dataset, in particular, the test split. See [nuPlan](../data/nuplan.md).

```bash
python -m py123d_garage.evaluation.navsim.evaluate \
    policy_config.evaluation_checkpoint_file=<checkpoint.pth> \
    parallelization_config.device=cuda
```

We parallelize the work with `parallelization_config.num_shards` and `parallelization_config.shard_index`. The sharding backend can be anything, from ray to SLURM array or simple GNU parallel.

## Other datasets

NAVSIM runs on any 123D dataset with 3D boxes and an HD map. Declare the dataset as a source yaml under `src/py123d_garage/config/presets/yaml/offline_data_sources/<policy>/`, then attach it as in [open loop](open_loop.md#other-datasets). One extra rule: the PDM score simulates 4 s plus 1 s of time-to-collision lookahead, so the source must declare at least 5 s of future. The shipped yamls declare 4 s, hence the override below, shown on the nuPlan val split:

```bash
python -m py123d_garage.evaluation.navsim.evaluate \
    policy_config.evaluation_checkpoint_file=<checkpoint.pth> \
    +offline_data_sources/ltf_nuplan@benchmark_offline_data_sources.nuplan_val=nuplan_val \
    benchmark_offline_data_sources.nuplan_val.cache_root=null \
    benchmark_offline_data_sources.nuplan_val.garage_scene_filter.future_duration_s=5.0 \
    parallelization_config.device=cuda
```
