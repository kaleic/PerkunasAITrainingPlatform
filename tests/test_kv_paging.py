from __future__ import annotations

import numpy as np

from kvserve.kv.compression import SelectiveCompressionConfig, SelectiveKVCompressor
from kvserve.kv.paging import KVPager, KVTier


def test_nvme_paging_round_trips_without_pickle(tmp_path) -> None:
    rng = np.random.default_rng(31)
    tensor = rng.standard_normal((1, 2, 24, 8), dtype=np.float32)
    compressor = SelectiveKVCompressor(model_seed=123)
    payload = compressor.compress(
        tensor,
        SelectiveCompressionConfig(recent_window=8, cold_bit_width=3, residual_ratio=0.001),
    )
    pager = KVPager(tmp_path)

    page = pager.put(
        tenant_id="tenant",
        model_id="model",
        request_id="request",
        payload=payload,
        token_start=0,
        token_end=24,
        tier=KVTier.NVME,
    )
    page.payload = None

    restored_payload = pager.get(page.page_id)
    restored = compressor.decompress(restored_payload)

    assert page.path is not None
    assert page.path.suffix == ".npz"
    assert list(tmp_path.glob("*.pkl")) == []
    assert restored.shape == tensor.shape
    assert len(restored_payload.segments) == len(payload.segments)
