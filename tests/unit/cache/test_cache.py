"""Unit tests for the cache store: codecs, the LMDB backend, manifest and signature."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from py123d_garage.cache._lmdb_backend import (
    LmdbCacheWriter,
    read_cached_tensor_addresses,
)
from py123d_garage.cache.build_cache import (
    _check_no_unsealed_leftovers,
    _manifest_from_shard_records,
    _record_finished_shard,
)
from py123d_garage.cache.cache_store import (
    CacheStoreReader,
    check_builder_cache_signature,
    check_cache_signature,
    delete_manifest,
    read_manifest,
    write_manifest,
)
from py123d_garage.cache.codec import (
    JpegCodec,
    PngCodec,
    ZlibCodec,
    decode_tensor,
    encode_tensor,
)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.randn(8),
        torch.zeros(30, dtype=torch.bool),
        torch.randn(4, 80, 96),
        torch.zeros(2, 80, 96),
        torch.randint(0, 12, (80, 96)),
    ],
)
def test_zlib_codec_is_lossless(tensor: torch.Tensor) -> None:
    """The mostly-zero label grids ride zlib; any dtype must round-trip exactly."""
    decoded = ZlibCodec().decode(ZlibCodec().encode(tensor))
    assert torch.equal(tensor, decoded)
    assert tensor.dtype == decoded.dtype


def test_zlib_codec_shrinks_sparse_rasters() -> None:
    sparse = torch.zeros(4, 80, 96)
    sparse[0, 3, 5] = 1.0
    assert len(ZlibCodec().encode(sparse)["b"]) < sparse.numel() * 4 // 100


def test_png_codec_preserves_class_ids() -> None:
    """A semantic map holds discrete class ids; PNG must return them exactly."""
    bev = torch.randint(0, 7, (128, 256)).float()
    codec = PngCodec(quantization_scale=1)
    decoded = codec.decode(codec.encode(bev))
    assert torch.equal(bev, decoded)
    assert decoded.dtype == torch.float32


def test_png_codec_preserves_normalized_raster() -> None:
    """A [0, 1] occupancy raster quantized to multiples of 1/5 survives 8-bit PNG."""
    lidar = torch.randint(0, 6, (1, 256, 256)).float() / 5
    codec = PngCodec(quantization_scale=255)
    decoded = codec.decode(codec.encode(lidar))
    assert torch.equal(lidar, decoded)
    assert decoded.shape == lidar.shape


def test_png_codec_refuses_lossy_quantization() -> None:
    """The guard that stops a wrong `quantization_scale` from silently corrupting cached tensors."""
    with pytest.raises(ValueError, match="not lossless"):
        PngCodec(quantization_scale=255).encode(
            torch.tensor([0.1234567, 0.7654321]),
        )


def test_jpeg_codec_restores_shape_and_dtype() -> None:
    camera = torch.rand(3, 64, 128)
    decoded = JpegCodec(quality=90).decode(JpegCodec(quality=90).encode(camera))
    assert decoded.shape == camera.shape
    assert decoded.dtype == camera.dtype


def test_jpeg_codec_would_corrupt_class_ids() -> None:
    """Why bev_semantic must be PNG: JPEG invents class ids that were never present."""
    bev = torch.randint(0, 7, (128, 256)).float()
    codec = JpegCodec(quality=95, quantization_scale=1)
    decoded = codec.decode(codec.encode(bev))
    assert not torch.equal(bev, decoded)


def test_encoded_record_decodes_without_the_writing_spec() -> None:
    """A record names its own codec, so decode needs no codec spec."""
    tensor = torch.randint(0, 6, (1, 32, 48)).float() / 5
    blob = encode_tensor(tensor, PngCodec(quantization_scale=255))
    assert torch.equal(decode_tensor(blob), tensor)


def test_store_round_trip_and_address_scan(tmp_path) -> None:
    """Per-tensor records under <sample_key>/<tensor_name>, one LMDB env per log."""
    camera = torch.rand(3, 8, 16)
    waypoints = torch.randn(8, 3)
    with LmdbCacheWriter(tmp_path, "log_a") as writer:
        writer.write(
            "uuid1/camera_feature",
            encode_tensor(camera, ZlibCodec()),
        )
        writer.write(
            "uuid1/trajectory",
            encode_tensor(waypoints, ZlibCodec()),
        )
    with LmdbCacheWriter(tmp_path, "log_b") as writer:
        writer.write(
            "uuid2/camera_feature",
            encode_tensor(camera, ZlibCodec()),
        )
    write_manifest(
        tmp_path,
        {
            "tensors": {
                "camera_feature": ZlibCodec().spec,
                "trajectory": ZlibCodec().spec,
            },
            "cache_signature": {
                "camera_feature": {},
                "trajectory": {},
            },
        },
    )

    assert read_cached_tensor_addresses(tmp_path, "log_a") == {
        "uuid1/camera_feature",
        "uuid1/trajectory",
    }
    assert read_cached_tensor_addresses(tmp_path, "missing_log") == set()

    reader = CacheStoreReader(
        tmp_path,
        {"camera_feature": {}, "trajectory": {}},
    )
    assert reader.tensor_names == {"camera_feature", "trajectory"}
    tensors = reader.read("log_a", "uuid1")
    assert torch.equal(tensors["camera_feature"], camera)
    assert torch.equal(tensors["trajectory"], waypoints)
    with pytest.raises(KeyError, match="rebuild"):
        reader.read("log_b", "uuid2")
    reader.close()

    # A consumer declaring fewer tensors reads only those.
    subset_reader = CacheStoreReader(tmp_path, {"camera_feature": {}})
    assert subset_reader.tensor_names == {"camera_feature"}
    assert set(subset_reader.read("log_b", "uuid2")) == {"camera_feature"}
    subset_reader.close()


def test_missing_sample_keys_reports_uncached_scenes(tmp_path) -> None:
    """The startup coverage check: selected-but-unstored samples surface at init."""
    with LmdbCacheWriter(tmp_path, "log_a") as writer:
        writer.write(
            "uuid1/camera_feature",
            encode_tensor(torch.rand(3, 8, 16), ZlibCodec()),
        )
    write_manifest(
        tmp_path,
        {
            "tensors": {"camera_feature": ZlibCodec().spec},
            "cache_signature": {"camera_feature": {}},
        },
    )
    reader = CacheStoreReader(tmp_path, {"camera_feature": {}})
    assert reader.missing_sample_keys("log_a", ["uuid1"]) == []
    assert reader.missing_sample_keys("log_a", ["uuid1", "uuid2"]) == ["uuid2"]
    assert reader.missing_sample_keys("never_built_log", ["uuid1"]) == ["uuid1"]
    reader.close()


def test_reader_refuses_missing_store(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="build it"):
        CacheStoreReader(tmp_path / "never_built", {})


def test_signature_mismatch_refuses_the_store(tmp_path) -> None:
    """A config change affecting cached content surfaces as a rebuild, not wrong tensors."""
    write_manifest(
        tmp_path,
        {
            "tensors": {},
            "cache_signature": {
                "lidar_feature": {"bev_pixels_per_meter": "4.0"},
            },
        },
    )
    check_cache_signature(
        read_manifest(tmp_path),
        {"lidar_feature": {"bev_pixels_per_meter": "4.0"}},
    )
    with pytest.raises(ValueError, match="bev_pixels_per_meter"):
        CacheStoreReader(
            tmp_path,
            {"lidar_feature": {"bev_pixels_per_meter": "2.0"}},
        )
    # A store that was never built is not a mismatch; the caller is about to build it.
    check_cache_signature(None, {"lidar_feature": {"anything": "goes"}})


def test_superset_store_serves_subset_consumer(tmp_path) -> None:
    """A store built with more heads on serves a consumer reading fewer tensors."""
    write_manifest(
        tmp_path,
        {
            "tensors": {},
            "cache_signature": {
                "lidar_feature": {"bev_pixels_per_meter": "4.0"},
                "trajectory": {"num_predicted_waypoints": "8"},
            },
        },
    )
    check_cache_signature(
        read_manifest(tmp_path),
        {"lidar_feature": {"bev_pixels_per_meter": "4.0"}},
    )
    with pytest.raises(ValueError, match="missing from the store"):
        check_cache_signature(
            read_manifest(tmp_path),
            {"semantic": {"image_width": "1152"}},
        )


def test_builder_refuses_narrowing_a_wider_store(tmp_path) -> None:
    """A narrower build must not reseal a wider store's manifest with fewer tensors."""
    write_manifest(
        tmp_path,
        {
            "tensors": {},
            "cache_signature": {
                "lidar_feature": {"bev_pixels_per_meter": "4.0"},
                "trajectory": {"num_predicted_waypoints": "8"},
            },
        },
    )
    check_builder_cache_signature(
        read_manifest(tmp_path),
        {
            "lidar_feature": {"bev_pixels_per_meter": "4.0"},
            "trajectory": {"num_predicted_waypoints": "8"},
        },
    )
    with pytest.raises(ValueError, match="also holds"):
        check_builder_cache_signature(
            read_manifest(tmp_path),
            {"lidar_feature": {"bev_pixels_per_meter": "4.0"}},
        )
    # A store that was never built is not a mismatch; the caller is about to build it.
    check_builder_cache_signature(None, {"lidar_feature": {"anything": "goes"}})


def test_empty_shard_records_and_seal_uses_a_sibling_manifest(tmp_path) -> None:
    """Empty shards complete the run; the seal takes a non-empty shard's manifest."""
    manifest = {
        "tensors": {},
        "cache_signature": {"lidar_feature": {"bev_pixels_per_meter": "4.0"}},
    }
    root = str(tmp_path)
    assert not _record_finished_shard(root, "run", 0, 3, None)
    assert not _record_finished_shard(root, "run", 1, 3, manifest)
    assert _record_finished_shard(root, "run", 2, 3, None)
    assert _manifest_from_shard_records(root, "run", 3) == manifest
    # A run whose every shard was empty has no manifest to seal with.
    empty_root = str(tmp_path / "all_empty")
    assert _record_finished_shard(empty_root, "run", 0, 1, None)
    assert _manifest_from_shard_records(empty_root, "run", 1) is None


def test_manifest_round_trip(tmp_path) -> None:
    manifest = {
        "tensors": {
            "camera_feature": JpegCodec(quality=90).spec,
            "trajectory": ZlibCodec().spec,
        },
        "cache_signature": {"camera_feature": {"image_width": "1024"}},
    }
    write_manifest(tmp_path, manifest)
    assert read_manifest(tmp_path) == manifest
    assert read_manifest(tmp_path / "nowhere") is None


def test_resume_refuses_unsealed_leftovers(tmp_path) -> None:
    root = str(tmp_path)
    _check_no_unsealed_leftovers(root, ["log_a", "log_b"])
    (tmp_path / "log_a").mkdir()
    with pytest.raises(RuntimeError, match="log_a"):
        _check_no_unsealed_leftovers(root, ["log_a", "log_b"])
    _check_no_unsealed_leftovers(root, ["log_b"])
    write_manifest(tmp_path, {"tensors": {}, "cache_signature": {}})
    _check_no_unsealed_leftovers(root, ["log_a", "log_b"])


def test_delete_manifest_unseals_the_store(tmp_path) -> None:
    write_manifest(tmp_path, {"tensors": {}, "cache_signature": {}})
    delete_manifest(tmp_path)
    assert read_manifest(tmp_path) is None
    delete_manifest(tmp_path)


def test_store_reader_survives_pickling(tmp_path) -> None:
    """DataLoader workers receive the reader through fork/pickle; envs must not travel."""
    import pickle

    with LmdbCacheWriter(tmp_path, "log_a") as writer:
        writer.write(
            "uuid1/velocity",
            encode_tensor(torch.tensor([1.0]), ZlibCodec()),
        )
    write_manifest(
        tmp_path,
        {
            "tensors": {"velocity": ZlibCodec().spec},
            "cache_signature": {"velocity": {}},
        },
    )

    reader = CacheStoreReader(tmp_path, {"velocity": {}})
    reader.read("log_a", "uuid1")  # opens the env
    cloned = pickle.loads(pickle.dumps(reader))
    tensors = cloned.read("log_a", "uuid1")
    assert np.allclose(tensors["velocity"].numpy(), [1.0])
    reader.close()
    cloned.close()
