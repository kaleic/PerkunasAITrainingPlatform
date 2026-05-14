from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import numpy as np

from kvserve.models.schemas import KVCompressionMode


@dataclass(frozen=True, slots=True)
class CompressionConfig:
    mode: KVCompressionMode = KVCompressionMode.TURBOQUANT
    bit_width: int = 4
    group_size: int = 64
    residual_ratio: float = 0.0
    rotation_seed: int = 17
    fp8_scale_floor: float = 1e-8

    def __post_init__(self) -> None:
        if self.bit_width > 4 or self.bit_width < 2:
            raise ValueError("TurboQuant bit_width must be between 2 and 4")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if not 0.0 <= self.residual_ratio <= 0.05:
            raise ValueError("residual_ratio must be in [0.0, 0.05]")


@dataclass(slots=True)
class CompressedTensor:
    mode: KVCompressionMode
    shape: tuple[int, ...]
    dtype: str
    payload: bytes
    metadata: dict[str, object] = field(default_factory=dict)
    logical_nbytes: int = 0

    @property
    def actual_nbytes(self) -> int:
        total = len(self.payload)
        for value in self.metadata.values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
            elif isinstance(value, (bytes, bytearray)):
                total += len(value)
            elif isinstance(value, (list, tuple)):
                total += sum(getattr(item, "nbytes", 0) for item in value)
        return total


def compress_tensor(array: np.ndarray, config: CompressionConfig) -> CompressedTensor:
    array = np.asarray(array)
    if config.mode == KVCompressionMode.STANDARD:
        return StandardCodec.compress(array)
    if config.mode == KVCompressionMode.FP8:
        return FP8E4M3Codec.compress(array, scale_floor=config.fp8_scale_floor)
    if config.mode == KVCompressionMode.TURBOQUANT:
        return TurboQuantCodec.compress(array, config)
    raise ValueError(f"unsupported compression mode: {config.mode}")


def decompress_tensor(tensor: CompressedTensor) -> np.ndarray:
    if tensor.mode == KVCompressionMode.STANDARD:
        return StandardCodec.decompress(tensor)
    if tensor.mode == KVCompressionMode.FP8:
        return FP8E4M3Codec.decompress(tensor)
    if tensor.mode == KVCompressionMode.TURBOQUANT:
        return TurboQuantCodec.decompress(tensor)
    raise ValueError(f"unsupported compression mode: {tensor.mode}")


class StandardCodec:
    @staticmethod
    def compress(array: np.ndarray) -> CompressedTensor:
        stored = np.asarray(array, dtype=np.float16)
        return CompressedTensor(
            mode=KVCompressionMode.STANDARD,
            shape=tuple(array.shape),
            dtype=str(stored.dtype),
            payload=stored.tobytes(order="C"),
            logical_nbytes=int(stored.nbytes),
        )

    @staticmethod
    def decompress(tensor: CompressedTensor) -> np.ndarray:
        return np.frombuffer(tensor.payload, dtype=np.dtype(tensor.dtype)).reshape(tensor.shape).copy()


class FP8E4M3Codec:
    BIAS = 7
    MAX_FINITE = (1.0 + 7.0 / 8.0) * (2.0 ** (15 - BIAS))

    @classmethod
    def compress(cls, array: np.ndarray, scale_floor: float = 1e-8) -> CompressedTensor:
        source = np.asarray(array, dtype=np.float32)
        max_abs = float(np.max(np.abs(source))) if source.size else 0.0
        scale = max(max_abs / cls.MAX_FINITE, scale_floor)
        encoded = cls.encode(source / scale)
        return CompressedTensor(
            mode=KVCompressionMode.FP8,
            shape=tuple(source.shape),
            dtype="float32",
            payload=encoded.tobytes(order="C"),
            metadata={"scale": np.float32(scale)},
            logical_nbytes=int(source.size),
        )

    @classmethod
    def decompress(cls, tensor: CompressedTensor) -> np.ndarray:
        encoded = np.frombuffer(tensor.payload, dtype=np.uint8)
        scale = float(tensor.metadata["scale"])
        decoded = cls.decode(encoded).reshape(tensor.shape)
        return (decoded * scale).astype(np.float32, copy=False)

    @classmethod
    def encode(cls, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        clipped = np.clip(values, -cls.MAX_FINITE, cls.MAX_FINITE)
        sign = (clipped < 0).astype(np.uint8)
        abs_values = np.abs(clipped)
        encoded = np.zeros(abs_values.shape, dtype=np.uint8)
        nonzero = abs_values > 0
        if not np.any(nonzero):
            return encoded

        normal_min = 2.0 ** (1 - cls.BIAS)
        normal = nonzero & (abs_values >= normal_min)
        subnormal = nonzero & ~normal

        if np.any(normal):
            exp_unbiased = np.floor(np.log2(abs_values[normal])).astype(np.int32)
            mantissa = np.round((abs_values[normal] / (2.0**exp_unbiased) - 1.0) * 8.0).astype(
                np.int32
            )
            carry = mantissa == 8
            exp_unbiased[carry] += 1
            mantissa[carry] = 0
            exp = np.clip(exp_unbiased + cls.BIAS, 1, 15)
            encoded[normal] = ((exp.astype(np.uint8) & 0x0F) << 3) | (
                mantissa.astype(np.uint8) & 0x07
            )

        if np.any(subnormal):
            mantissa = np.round(abs_values[subnormal] / normal_min * 8.0).astype(np.int32)
            mantissa = np.clip(mantissa, 0, 7)
            encoded[subnormal] = mantissa.astype(np.uint8)

        encoded |= sign << 7
        return encoded

    @classmethod
    def decode(cls, encoded: np.ndarray) -> np.ndarray:
        encoded = np.asarray(encoded, dtype=np.uint8)
        sign = np.where((encoded & 0x80) != 0, -1.0, 1.0).astype(np.float32)
        exp = ((encoded >> 3) & 0x0F).astype(np.int32)
        mantissa = (encoded & 0x07).astype(np.float32)
        normal_min = 2.0 ** (1 - cls.BIAS)
        normal = exp > 0
        values = np.zeros(encoded.shape, dtype=np.float32)
        values[normal] = (1.0 + mantissa[normal] / 8.0) * (2.0 ** (exp[normal] - cls.BIAS))
        values[~normal] = (mantissa[~normal] / 8.0) * normal_min
        return values * sign


class TurboQuantCodec:
    @classmethod
    def compress(cls, array: np.ndarray, config: CompressionConfig) -> CompressedTensor:
        source = np.asarray(array, dtype=np.float32)
        original_shape = tuple(source.shape)
        if source.ndim == 0:
            raise ValueError("cannot compress scalar KV tensor")
        dim = source.shape[-1]
        rows = int(source.size // dim)
        matrix = source.reshape(rows, dim)
        rotation = rotation_matrix(dim, config.rotation_seed)
        rotated = matrix @ rotation

        padded_dim = int(math.ceil(dim / config.group_size) * config.group_size)
        if padded_dim != dim:
            padded = np.zeros((rows, padded_dim), dtype=np.float32)
            padded[:, :dim] = rotated
        else:
            padded = rotated

        qmax = (2 ** (config.bit_width - 1)) - 1
        groups = padded_dim // config.group_size
        scales = np.empty((rows, groups), dtype=np.float16)
        quantized = np.empty((rows, padded_dim), dtype=np.int16)

        for group_index in range(groups):
            start = group_index * config.group_size
            stop = start + config.group_size
            block = padded[:, start:stop]
            max_abs = np.max(np.abs(block), axis=1)
            scale = np.maximum(max_abs / qmax, 1e-8)
            scales[:, group_index] = scale.astype(np.float16)
            quantized[:, start:stop] = np.clip(np.round(block / scale[:, None]), -qmax, qmax).astype(
                np.int16
            )

        packed = pack_signed_nbit(quantized.reshape(-1), config.bit_width, qmax)
        metadata: dict[str, object] = {
            "bit_width": np.uint8(config.bit_width),
            "group_size": np.uint16(config.group_size),
            "rows": np.uint32(rows),
            "dim": np.uint32(dim),
            "padded_dim": np.uint32(padded_dim),
            "scales": scales,
            "rotation_seed": np.uint32(config.rotation_seed),
        }

        tensor = CompressedTensor(
            mode=KVCompressionMode.TURBOQUANT,
            shape=original_shape,
            dtype="float32",
            payload=packed,
            metadata=metadata,
            logical_nbytes=int(math.ceil(quantized.size * config.bit_width / 8) + scales.nbytes),
        )

        if config.residual_ratio > 0.0 and source.size:
            reconstructed = cls.decompress(tensor)
            residual = (source - reconstructed).reshape(-1)
            keep = max(1, int(source.size * config.residual_ratio))
            indices = np.argpartition(np.abs(residual), -keep)[-keep:].astype(np.uint32)
            values = residual[indices].astype(np.float16)
            tensor.metadata["residual_indices"] = indices
            tensor.metadata["residual_values"] = values
            tensor.logical_nbytes += int(indices.nbytes + values.nbytes)

        return tensor

    @classmethod
    def decompress(cls, tensor: CompressedTensor) -> np.ndarray:
        bit_width = int(tensor.metadata["bit_width"])
        group_size = int(tensor.metadata["group_size"])
        rows = int(tensor.metadata["rows"])
        dim = int(tensor.metadata["dim"])
        padded_dim = int(tensor.metadata["padded_dim"])
        qmax = (2 ** (bit_width - 1)) - 1
        count = rows * padded_dim
        quantized = unpack_signed_nbit(tensor.payload, count, bit_width, qmax).reshape(
            rows, padded_dim
        )
        scales = np.asarray(tensor.metadata["scales"], dtype=np.float32)
        groups = padded_dim // group_size
        restored = np.empty((rows, padded_dim), dtype=np.float32)
        for group_index in range(groups):
            start = group_index * group_size
            stop = start + group_size
            restored[:, start:stop] = quantized[:, start:stop].astype(np.float32) * scales[
                :, group_index, None
            ]
        restored = restored[:, :dim]
        rotation = rotation_matrix(dim, int(tensor.metadata["rotation_seed"]))
        matrix = restored @ rotation.T
        output = matrix.reshape(tensor.shape)
        if "residual_indices" in tensor.metadata:
            flat = output.reshape(-1)
            indices = np.asarray(tensor.metadata["residual_indices"], dtype=np.uint32)
            values = np.asarray(tensor.metadata["residual_values"], dtype=np.float16)
            flat[indices] += values.astype(np.float32)
        return output.astype(np.float32, copy=False)


@lru_cache(maxsize=64)
def rotation_matrix(dim: int, seed: int) -> np.ndarray:
    seed_material = hashlib.sha256(f"{dim}:{seed}".encode("ascii")).digest()
    int_seed = int.from_bytes(seed_material[:8], "little", signed=False) % (2**32)
    rng = np.random.default_rng(int_seed)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs
    return q.astype(np.float32)


def pack_signed_nbit(values: np.ndarray, bit_width: int, qmax: int) -> bytes:
    unsigned = (np.asarray(values, dtype=np.int16) + qmax).astype(np.uint8)
    max_code = (1 << bit_width) - 1
    if np.any(unsigned > max_code):
        raise ValueError("quantized value outside packable range")
    out = bytearray(math.ceil(unsigned.size * bit_width / 8))
    bit_position = 0
    for code in unsigned:
        byte_index = bit_position // 8
        offset = bit_position % 8
        value = int(code) << offset
        out[byte_index] |= value & 0xFF
        if offset + bit_width > 8:
            out[byte_index + 1] |= (value >> 8) & 0xFF
        bit_position += bit_width
    return bytes(out)


def unpack_signed_nbit(payload: bytes, count: int, bit_width: int, qmax: int) -> np.ndarray:
    data = np.frombuffer(payload, dtype=np.uint8)
    mask = (1 << bit_width) - 1
    values = np.empty(count, dtype=np.int16)
    bit_position = 0
    for index in range(count):
        byte_index = bit_position // 8
        offset = bit_position % 8
        raw = int(data[byte_index]) >> offset
        if offset + bit_width > 8 and byte_index + 1 < data.size:
            raw |= int(data[byte_index + 1]) << (8 - offset)
        values[index] = (raw & mask) - qmax
        bit_position += bit_width
    return values


@dataclass(slots=True)
class CompressedSegment:
    indices: np.ndarray
    tensor: CompressedTensor
    precision_tier: Literal["recent", "hot", "cold"]


@dataclass(slots=True)
class SelectiveCompressedTensor:
    shape: tuple[int, ...]
    token_axis: int
    segments: list[CompressedSegment]

    @property
    def actual_nbytes(self) -> int:
        return sum(segment.tensor.actual_nbytes + segment.indices.nbytes for segment in self.segments)

    @property
    def logical_nbytes(self) -> int:
        return sum(segment.tensor.logical_nbytes + segment.indices.nbytes for segment in self.segments)


@dataclass(frozen=True, slots=True)
class SelectiveCompressionConfig:
    recent_window: int = 256
    high_attention_fraction: float = 0.10
    cold_bit_width: int = 3
    hot_mode: KVCompressionMode = KVCompressionMode.FP8
    token_axis: int = -2
    residual_ratio: float = 0.002


class SelectiveKVCompressor:
    def __init__(self, model_seed: int):
        self.model_seed = model_seed

    def compress(
        self,
        array: np.ndarray,
        config: SelectiveCompressionConfig,
        attention_scores: np.ndarray | None = None,
    ) -> SelectiveCompressedTensor:
        source = np.asarray(array, dtype=np.float32)
        token_axis = config.token_axis if config.token_axis >= 0 else source.ndim + config.token_axis
        token_count = source.shape[token_axis]
        moved = np.moveaxis(source, token_axis, 0)
        recent_count = min(config.recent_window, token_count)
        recent_indices = np.arange(token_count - recent_count, token_count, dtype=np.int32)
        old_indices = np.arange(0, token_count - recent_count, dtype=np.int32)

        hot_indices = np.array([], dtype=np.int32)
        if old_indices.size:
            if attention_scores is None:
                scores = np.linspace(0.0, 1.0, token_count, dtype=np.float32)
            else:
                scores = np.asarray(attention_scores, dtype=np.float32).reshape(token_count)
            hot_count = min(old_indices.size, int(math.ceil(token_count * config.high_attention_fraction)))
            if hot_count > 0:
                old_scores = scores[old_indices]
                hot_positions = np.argpartition(old_scores, -hot_count)[-hot_count:]
                hot_indices = np.sort(old_indices[hot_positions].astype(np.int32))
        cold_indices = np.setdiff1d(old_indices, hot_indices, assume_unique=True).astype(np.int32)

        segments: list[CompressedSegment] = []
        if cold_indices.size:
            cold_tensor = compress_tensor(
                moved[cold_indices],
                CompressionConfig(
                    mode=KVCompressionMode.TURBOQUANT,
                    bit_width=config.cold_bit_width,
                    residual_ratio=config.residual_ratio,
                    rotation_seed=self.model_seed,
                ),
            )
            segments.append(CompressedSegment(cold_indices, cold_tensor, "cold"))
        if hot_indices.size:
            hot_tensor = compress_tensor(moved[hot_indices], CompressionConfig(mode=config.hot_mode))
            segments.append(CompressedSegment(hot_indices, hot_tensor, "hot"))
        if recent_indices.size:
            recent_tensor = compress_tensor(
                moved[recent_indices], CompressionConfig(mode=KVCompressionMode.STANDARD)
            )
            segments.append(CompressedSegment(recent_indices, recent_tensor, "recent"))
        return SelectiveCompressedTensor(shape=tuple(source.shape), token_axis=token_axis, segments=segments)

    def decompress(self, tensor: SelectiveCompressedTensor) -> np.ndarray:
        moved_shape = list(tensor.shape)
        token_axis = tensor.token_axis
        token_count = moved_shape[token_axis]
        moved_shape.pop(token_axis)
        target = np.zeros((token_count, *moved_shape), dtype=np.float32)
        for segment in tensor.segments:
            target[segment.indices] = decompress_tensor(segment.tensor)
        return np.moveaxis(target, 0, token_axis)
