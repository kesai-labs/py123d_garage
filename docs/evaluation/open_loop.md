<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">Open-loop evaluation</h1>

Open loop compares the planned trajectory with the logged one, as displacement errors. It needs only an offline dataset; the default is the nuPlan navtest split. By default the script scores seed 0 of the newest posttrain run; set `CHECKPOINT_FILE` for another one.

```bash
scripts/evaluation/open_loop.sh
```

To split the work over several machines, pass `parallelization_config.num_shards` and `parallelization_config.shard_index`.
