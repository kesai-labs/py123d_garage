<div align="center">

<img src="../assets/logo.png" alt="Py123d Garage" width="630">

</div>

<h1 align="center">NAVSIM evaluation</h1>

py123d_garage evaluates on the navtest split of [NAVSIM](https://github.com/autonomousvision/navsim) in two ways: with its own port of the PDM score, to compare checkpoints with each other, and by exporting the planned trajectories and scoring them with the official devkits, for numbers comparable to published results. Both need the nuPlan test split, see [nuPlan](../data/nuplan.md).

## PDM score port

> \[!CAUTION\]
> The port is a reimplementation, not the official NAVSIM. The PDM score is rewritten on 123D, and the policy is conditioned on target points instead of the driving command. Its scores are not aligned with the official devkits or the NAVSIM leaderboard and are only comparable with each other.

The port simulates the predicted trajectory for 4 seconds and scores collisions, drivable area, driving direction, time to collision, progress and comfort.

Every navtest frame whose log covers the 5 s scoring window is scored, whatever the policy's plan horizon. A frame whose route ends before the policy's farthest target point is scored too: the points past the route's end are served as NaN, which the policy reads as no navigation intent. A scene the scorer cannot handle fails the run after `results.csv` is written; its row carries the error in the `scoring_error` column.

By default the script scores seed 0 of the newest posttrain run; set `CHECKPOINT_FILE` for another one.

```bash
scripts/evaluation/navsim.sh
```

To split the work over several machines, pass `parallelization_config.num_shards` and `parallelization_config.shard_index`.

## Official scoring

With `save_trajectories=true`, the run also writes the planned trajectory of every scene to `trajectories/` in its run directory, with the scene's log name and timestamp and the poses relative to the ego.

```bash
scripts/evaluation/navsim.sh save_trajectories=true
```

The official devkits score these trajectories once they are converted into a NAVSIM submission: one trajectory per navtest token, sampled at the 8 poses 0.5 s apart that NAVSIM scores, with the token found from the log name and timestamp. The few navtest frames the port skips have no trajectory and need a fallback. NAVSIM v1.1's `run_pdm_score_from_submission.py` gives the PDMS. Only the scoring is official: the policy keeps its own inputs, not NAVSIM's driving command. The navtest table of the README is scored this way.

The navhard two-stage table of the README is scored the same way, with the two-stage runner of NAVSIM v2.2. The real frames of the first stage are planned on 123D like navtest, and the synthetic frames of the second stage, which have no 123D conversion, are read straight from NAVSIM's scene files. That reader is not part of py123d_garage.
