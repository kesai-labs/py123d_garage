from __future__ import annotations

import io
import pickle
import zlib
from collections.abc import Mapping
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from py123d.common.io.camera.jpeg_camera_io import decode_image_from_jpeg_binary
from py123d.common.io.camera.png_camera_io import decode_image_from_png_binary
from typing_extensions import override

# Keys inside an encoded record, kept short because they repeat once per stored tensor.
# The values persist in every built store -- never change them.
_CODEC: Final[str] = "c"
_DTYPE: Final[str] = "d"
_SHAPE: Final[str] = "s"
_BYTES: Final[str] = "b"
_PARAMS: Final[str] = "p"


class TensorCodec:
    """Encodes a single tensor to bytes and back."""

    name: str

    @property
    def params(self) -> dict[str, Any]:
        """The codec's constructor keyword arguments."""
        return {}

    @property
    def spec(self) -> dict[str, Any]:
        """The codec's name and parameters as plain data, for the store manifest."""
        return {"name": self.name, **self.params}

    def encode(self, tensor: torch.Tensor) -> dict[str, Any]:
        """
        Encode one tensor.

        Args:
            tensor: the tensor to encode

        Returns:
            a self-describing entry holding the tensor's bytes and metadata
        """
        raise NotImplementedError

    def decode(self, entry: Mapping[str, Any]) -> torch.Tensor:
        """
        Decode one entry produced by encode.

        Args:
            entry: the entry to decode

        Returns:
            the reconstructed tensor
        """
        raise NotImplementedError


def _dtype_of(entry: Mapping[str, Any]) -> np.dtype[np.generic]:
    return cast(np.dtype[np.generic], np.dtype(entry[_DTYPE]))


def _tensor_to_array(tensor: torch.Tensor) -> npt.NDArray[np.generic]:
    return cast(
        npt.NDArray[np.generic],
        tensor.detach().cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
    )


class ZlibCodec(TensorCodec):
    """zlib-compressed raw buffer. Lossless for every dtype; shrinks sparse arrays."""

    name = "zlib"

    def __init__(self, level: int = 1) -> None:
        self._compression_level = int(level)

    @property
    @override
    def params(self) -> dict[str, Any]:
        return {"level": self._compression_level}

    @override
    def encode(self, tensor: torch.Tensor) -> dict[str, Any]:
        array = _tensor_to_array(tensor)
        return {
            _CODEC: self.name,
            _DTYPE: array.dtype.str,
            _SHAPE: tuple(array.shape),
            _BYTES: zlib.compress(
                np.ascontiguousarray(array).tobytes(),
                self._compression_level,
            ),
        }

    @override
    def decode(self, entry: Mapping[str, Any]) -> torch.Tensor:
        # np.frombuffer returns a read-only view; torch needs it writable, hence the copy.
        array = np.frombuffer(
            zlib.decompress(entry[_BYTES]),
            dtype=_dtype_of(entry),
        ).reshape(entry[_SHAPE])
        return torch.as_tensor(array.copy())


def _to_image_array(array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Reshape a (H, W), (1, H, W) or (3, H, W) array into what PIL expects."""
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 3 and array.shape[0] == 3:
        return np.transpose(array, (1, 2, 0))
    raise ValueError(
        f"image codecs need (H, W), (1, H, W) or (3, H, W); got {array.shape}",
    )


def _from_image_array(
    image: npt.NDArray[np.uint8],
    shape: tuple[int, ...],
) -> npt.NDArray[np.uint8]:
    """
    Invert _to_image_array back to the tensor's original shape.

    A decoder may hand back a grayscale image expanded to three identical channels,
    so a single-channel tensor keeps only the first.
    """
    channels = shape[0] if len(shape) == 3 else 1
    if image.ndim == 3 and channels == 1:
        image = image[..., 0]
    if len(shape) == 3 and shape[0] == 3:
        image = np.transpose(image, (2, 0, 1))
    return image.reshape(shape)


class _ImageCodec(TensorCodec):
    """
    Shared 8-bit value quantization: u8 = round(value * quantization_scale).

    Maps the tensor's values onto uint8 pixels, never its spatial size:
    quantization_scale=255 for a tensor normalized to [0, 1];
    quantization_scale=1 for one already holding uint8-ranged values.
    """

    def __init__(self, quantization_scale: float = 255.0) -> None:
        self._quantization_scale = float(quantization_scale)

    @property
    @override
    def params(self) -> dict[str, Any]:
        return {"quantization_scale": self._quantization_scale}

    def _quantize(
        self,
        tensor: torch.Tensor,
    ) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.uint8]]:
        array = _tensor_to_array(tensor)
        quantized = np.clip(
            np.rint(array.astype(np.float64) * self._quantization_scale),
            0,
            255,
        ).astype(np.uint8)
        return array, quantized

    def _dequantize(
        self,
        image: npt.NDArray[np.uint8],
        entry: Mapping[str, Any],
    ) -> torch.Tensor:
        quantization_scale = entry[_PARAMS]["quantization_scale"]
        array: npt.NDArray[np.generic] = _from_image_array(
            image,
            tuple(entry[_SHAPE]),
        )
        if quantization_scale != 1.0:
            array = array.astype(np.float64) / quantization_scale
        return torch.as_tensor(array.astype(_dtype_of(entry)))


class PngCodec(_ImageCodec):
    """Lossless PNG. Raises on tensors that 8-bit quantization would not round-trip exactly."""

    name = "png"

    @override
    def encode(self, tensor: torch.Tensor) -> dict[str, Any]:
        array, quantized = self._quantize(tensor)
        restored = (
            quantized.astype(np.float64) / self._quantization_scale if self._quantization_scale != 1.0 else quantized
        )
        if not np.allclose(restored.astype(array.dtype), array, rtol=0, atol=0):
            raise ValueError(
                f"PngCodec(quantization_scale={self._quantization_scale}) is not lossless for this tensor "
                f"(dtype={array.dtype}, range=[{array.min()}, {array.max()}]). "
                f"Use quantization_scale=1 for integer-valued tensors, 255 for [0, 1] tensors, "
                f"or fall back to zlib.",
            )
        buffer = io.BytesIO()
        Image.fromarray(_to_image_array(quantized)).save(
            buffer,
            format="PNG",
            optimize=False,
        )
        return {
            _CODEC: self.name,
            _DTYPE: array.dtype.str,
            _SHAPE: tuple(array.shape),
            _BYTES: buffer.getvalue(),
            _PARAMS: self.params,
        }

    @override
    def decode(self, entry: Mapping[str, Any]) -> torch.Tensor:
        return self._dequantize(
            decode_image_from_png_binary(entry[_BYTES]),
            entry,
        )


class JpegCodec(_ImageCodec):
    """Lossy JPEG, for natural images only."""

    name = "jpeg"

    def __init__(
        self,
        quality: int = 90,
        quantization_scale: float = 255.0,
    ) -> None:
        super().__init__(quantization_scale=quantization_scale)
        self._quality = int(quality)

    @property
    @override
    def params(self) -> dict[str, Any]:
        return super().params | {"quality": self._quality}

    @override
    def encode(self, tensor: torch.Tensor) -> dict[str, Any]:
        array, quantized = self._quantize(tensor)
        buffer = io.BytesIO()
        Image.fromarray(_to_image_array(quantized)).save(
            buffer,
            format="JPEG",
            quality=self._quality,
        )
        return {
            _CODEC: self.name,
            _DTYPE: array.dtype.str,
            _SHAPE: tuple(array.shape),
            _BYTES: buffer.getvalue(),
            _PARAMS: self.params,
        }

    @override
    def decode(self, entry: Mapping[str, Any]) -> torch.Tensor:
        # py123d's reader, so a faster decoder installed via its set_jpeg_decoder also
        # serves the cache. Returns RGB, the order the encoder wrote.
        return self._dequantize(
            decode_image_from_jpeg_binary(entry[_BYTES]),
            entry,
        )


_CODEC_TYPES: Final[dict[str, type[TensorCodec]]] = {
    ZlibCodec.name: ZlibCodec,
    PngCodec.name: PngCodec,
    JpegCodec.name: JpegCodec,
}


def encode_tensor(tensor: torch.Tensor, codec: TensorCodec) -> bytes:
    """
    Encode one sample tensor into its stored record.

    Args:
        tensor: the tensor to store
        codec: the codec to encode it with, e.g. PngCodec(quantization_scale=1)

    Returns:
        the pickled record
    """
    return pickle.dumps(
        codec.encode(tensor),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def decode_tensor(blob: bytes) -> torch.Tensor:
    """
    Rebuild one stored tensor. The record names its own codec and parameters.

    Args:
        blob: the pickled record

    Returns:
        the reconstructed tensor
    """
    entry = pickle.loads(blob)
    codec_name = entry[_CODEC]
    if codec_name not in _CODEC_TYPES:
        raise ValueError(
            f"record uses unknown codec {codec_name!r}; available: {sorted(_CODEC_TYPES)}",
        )
    # Codecs are stateless at decode time: every parameter lives in the entry.
    codec = _CODEC_TYPES[codec_name](**entry.get(_PARAMS, {}))
    return codec.decode(entry)
