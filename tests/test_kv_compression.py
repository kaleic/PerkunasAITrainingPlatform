from __future__ import annotations

import numpy as np

from kvserve.kv.compression import (
    CompressionConfig,
    KVCompressionMode,
    SelectiveCompressionConfig,
    SelectiveKVCompressor,
    compress_tensor,
    decompress_tensor,
)
from kvserve.kv.pruning import prune_kv_tokens


def test_turboquant_reconstructs_with_bounded_error() -> None:
    rng = np.random.default_rng(7)
    tensor = rng.standard_normal((2, 3, 64, 16), dtype=np.float32)
    compressed = compress_tensor(
        tensor,
        CompressionConfig(
            mode=KVCompressionMode.TURBOQUANT,
            bit_width=4,
            group_size=16,
            residual_ratio=0.002,
        ),
    )
    restored = decompress_tensor(compressed)
    relative_error = np.linalg.norm(tensor - restored) / np.linalg.norm(tensor)
    assert relative_error < 0.22
    assert compressed.actual_nbytes < tensor.nbytes


def test_selective_compression_preserves_shape_and_saves_memory() -> None:
    rng = np.random.default_rng(11)
    tensor = rng.standard_normal((2, 4, 128, 32), dtype=np.float32)
    attention = np.linspace(0, 1, 128, dtype=np.float32)
    compressor = SelectiveKVCompressor(model_seed=123)
    compressed = compressor.compress(
        tensor,
        SelectiveCompressionConfig(recent_window=16, cold_bit_width=3, residual_ratio=0.001),
        attention,
    )
    restored = compressor.decompress(compressed)
    assert restored.shape == tensor.shape
    assert compressed.actual_nbytes < tensor.nbytes


def test_pruning_keeps_recent_tokens() -> None:
    rng = np.random.default_rng(21)
    key = rng.standard_normal((1, 2, 20, 8), dtype=np.float32)
    value = rng.standard_normal((1, 2, 20, 8), dtype=np.float32)
    scores = np.zeros(20, dtype=np.float32)
    result = prune_kv_tokens(key, value, 0.25, scores, recent_window=5)
    assert result.pruned_tokens == 5
    assert set(range(15, 20)).issubset(set(result.retained_indices.tolist()))
