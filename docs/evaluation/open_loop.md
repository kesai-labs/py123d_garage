<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/logo_dark.png">
  <img src="../assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">Open-loop evaluation</h1>

This evaluation protocol only requires an offline dataset.

## Default usage on nuPlan navtest

By default we use the navtest split:

```bash
python -m py123d_garage.evaluation.open_loop.evaluate \
    policy_config.evaluation_checkpoint_file=<checkpoint.pth> \
    parallelization_config.device=cuda
```

We parallelize the work with `parallelization_config.num_shards` and `parallelization_config.shard_index`. The sharding backend can be anything, from ray to SLURM array or simple GNU parallel.

## Other datasets

For example, we evaluate a checkpoint on nuPlan test split with the config `src/py123d_garage/config/presets/yaml/offline_data_sources/tf_nuplan/nuplan_test.yaml` and the command:

```bash
python -m py123d_garage.evaluation.open_loop.evaluate \
    policy_config.evaluation_checkpoint_file=<checkpoint.pth> \
    +offline_data_sources/ltf_nuplan@benchmark_offline_data_sources.nuplan_test=nuplan_test \
    benchmark_offline_data_sources.nuplan_test.cache_root=null \
    parallelization_config.device=cuda
```
