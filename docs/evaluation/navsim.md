<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">NAVSIM evaluation</h1>

> \[!CAUTION\]
> This is not the official NAVSIM but a reimplementation. The PDM score is rewritten on 123D, and the policy is conditioned on target points instead of the driving command. Scores are not comparable to the NAVSIM leaderboard.

NAVSIM scores a checkpoint with the PDM score on the navtest split. It simulates the predicted trajectory for 4 seconds and scores collisions, drivable area, driving direction, time to collision, progress and comfort.

Every navtest frame whose log covers the 5 s scoring window is scored, whatever the policy's plan horizon. A frame whose route ends before the policy's farthest target point is scored too: the points past the route's end are served as NaN, which the policy reads as no navigation intent. A scene the scorer cannot handle fails the run after `results.csv` is written; its row carries the error in the `scoring_error` column.

It needs the nuPlan test split, see [nuPlan](../data/nuplan.md). By default the script scores seed 0 of the newest posttrain run; set `CHECKPOINT_FILE` for another one.

```bash
scripts/evaluation/navsim.sh
```

To split the work over several machines, pass `parallelization_config.num_shards` and `parallelization_config.shard_index`.
