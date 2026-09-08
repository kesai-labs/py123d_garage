<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.png">
  <img src="assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">Training Cache System</h1>

The cache is optional. Set `read_from_cache_store=false` and you can load the training data directly from 123D logs..

Build it with the script of your dataset and model, then train as usual:

```bash
scripts/cache/<dataset>/build_<model>_cache.sh
```

Some details:

- Storage is one LMDB database per log under `<dataset>/py123d_garage_cache/<store>/`.
- Tensors are compressed. The policy sets the codec per tensor in `cache_codecs`: JPEG for images, PNG for quantized maps, zlib otherwise.
- The manifest records the policy config the store was built with. After changing the policy or a cache builder, rebuild with `force_cache_rebuild=true`.
- `persistent_workers=false` in the data loader, otherwise the workers run out of memory.
