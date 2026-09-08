<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.png">
  <img src="assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">Training a policy</h1>

Build the feature cache first. Training reads tensors from the cache instead of decoding the raw logs, which is several times faster per sample.

```bash
scripts/cache/<dataset>/build_<model>_cache.sh
```

The cache signature does not track the builder code. Rebuild by hand after you change a cache builder.

TransFuser trains in two stages. Perception pretraining:

```bash
scripts/training/<dataset>/<model>/pretrain.sh
```

Then planning posttraining, which starts from the pretrained weights. Point `initial_weights_file` in the script at the checkpoint of your pretraining run.

```bash
scripts/training/<dataset>/<model>/posttrain.sh
```

The scripts run on a single host and under SLURM. They derive the per-GPU batch size from the visible devices.
