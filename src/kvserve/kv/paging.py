from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from kvserve.kv.compression import CompressedSegment, CompressedTensor, SelectiveCompressedTensor
from kvserve.models.schemas import KVCompressionMode


class KVTier(StrEnum):
    GPU = "gpu"
    CPU = "cpu"
    NVME = "nvme"


@dataclass(slots=True)
class KVPage:
    page_id: str
    tenant_id: str
    model_id: str
    request_id: str
    tier: KVTier
    token_start: int
    token_end: int
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    payload: SelectiveCompressedTensor | None = None
    path: Path | None = None


class KVPager:
    def __init__(self, nvme_dir: Path):
        self.nvme_dir = nvme_dir
        self.nvme_dir.mkdir(parents=True, exist_ok=True)
        self.pages: dict[str, KVPage] = {}

    def put(
        self,
        tenant_id: str,
        model_id: str,
        request_id: str,
        payload: SelectiveCompressedTensor,
        token_start: int,
        token_end: int,
        tier: KVTier = KVTier.GPU,
    ) -> KVPage:
        page_id = f"kvp_{uuid.uuid4().hex}"
        page = KVPage(
            page_id=page_id,
            tenant_id=tenant_id,
            model_id=model_id,
            request_id=request_id,
            tier=tier,
            token_start=token_start,
            token_end=token_end,
            size_bytes=payload.actual_nbytes,
            payload=payload,
        )
        self.pages[page_id] = page
        if tier == KVTier.NVME:
            self._write_to_nvme(page)
        return page

    def get(self, page_id: str) -> SelectiveCompressedTensor:
        page = self.pages[page_id]
        page.last_accessed_at = time.time()
        if page.tier == KVTier.NVME:
            if page.payload is None:
                page.payload = self._read_from_nvme(page)
        if page.payload is None:
            raise RuntimeError(f"page {page_id} has no resident payload")
        return page.payload

    def move(self, page_id: str, tier: KVTier) -> KVPage:
        page = self.pages[page_id]
        if page.tier == tier:
            return page
        if tier == KVTier.NVME:
            self._write_to_nvme(page)
            page.payload = None
        elif page.tier == KVTier.NVME and page.payload is None:
            page.payload = self._read_from_nvme(page)
        page.tier = tier
        page.last_accessed_at = time.time()
        return page

    def evict_lru(self, tier: KVTier, bytes_needed: int) -> list[KVPage]:
        candidates = sorted(
            [page for page in self.pages.values() if page.tier == tier],
            key=lambda page: page.last_accessed_at,
        )
        evicted: list[KVPage] = []
        freed = 0
        for page in candidates:
            if freed >= bytes_needed:
                break
            if tier == KVTier.GPU:
                self.move(page.page_id, KVTier.CPU)
            elif tier == KVTier.CPU:
                self.move(page.page_id, KVTier.NVME)
            else:
                self.delete(page.page_id)
            freed += page.size_bytes
            evicted.append(page)
        return evicted

    def prefetch(self, request_id: str, token_position: int, window_tokens: int) -> list[KVPage]:
        selected: list[KVPage] = []
        for page in self.pages.values():
            if page.request_id != request_id:
                continue
            if page.token_start <= token_position + window_tokens and page.token_end >= token_position:
                if page.tier != KVTier.GPU:
                    self.move(page.page_id, KVTier.GPU)
                selected.append(page)
        return selected

    def delete(self, page_id: str) -> None:
        page = self.pages.pop(page_id)
        if page.path and page.path.exists():
            page.path.unlink()

    def tier_bytes(self, tier: KVTier) -> int:
        return sum(page.size_bytes for page in self.pages.values() if page.tier == tier)

    def _write_to_nvme(self, page: KVPage) -> None:
        if page.payload is None:
            return
        path = self.nvme_dir / f"{page.page_id}.npz"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, **_serialize_selective_tensor(page.payload))
        os.replace(tmp, path)
        page.path = path

    def _read_from_nvme(self, page: KVPage) -> SelectiveCompressedTensor:
        if page.path is None:
            raise RuntimeError(f"page {page.page_id} has no NVMe path")
        with np.load(page.path, allow_pickle=False) as archive:
            return _deserialize_selective_tensor(archive)


def _serialize_selective_tensor(tensor: SelectiveCompressedTensor) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, object] = {
        "shape": list(tensor.shape),
        "token_axis": tensor.token_axis,
        "segments": [],
    }
    segments = manifest["segments"]
    if not isinstance(segments, list):
        raise TypeError("segments manifest must be a list")

    for segment_index, segment in enumerate(tensor.segments):
        prefix = f"segment_{segment_index}"
        arrays[f"{prefix}_indices"] = np.asarray(segment.indices)
        arrays[f"{prefix}_payload"] = np.frombuffer(segment.tensor.payload, dtype=np.uint8).copy()
        metadata_manifest, metadata_arrays = _serialize_metadata(
            segment.tensor.metadata, f"{prefix}_metadata"
        )
        arrays.update(metadata_arrays)
        segments.append(
            {
                "indices": f"{prefix}_indices",
                "precision_tier": segment.precision_tier,
                "tensor": {
                    "mode": str(segment.tensor.mode),
                    "shape": list(segment.tensor.shape),
                    "dtype": segment.tensor.dtype,
                    "payload": f"{prefix}_payload",
                    "metadata": metadata_manifest,
                    "logical_nbytes": segment.tensor.logical_nbytes,
                },
            }
        )

    arrays["manifest"] = np.frombuffer(
        json.dumps(manifest, separators=(",", ":")).encode("utf-8"), dtype=np.uint8
    ).copy()
    return arrays


def _deserialize_selective_tensor(archive: np.lib.npyio.NpzFile) -> SelectiveCompressedTensor:
    manifest = json.loads(bytes(np.asarray(archive["manifest"], dtype=np.uint8)).decode("utf-8"))
    segments = []
    for segment_manifest in manifest["segments"]:
        tensor_manifest = segment_manifest["tensor"]
        metadata = _deserialize_metadata(tensor_manifest["metadata"], archive)
        tensor = CompressedTensor(
            mode=KVCompressionMode(tensor_manifest["mode"]),
            shape=tuple(int(dim) for dim in tensor_manifest["shape"]),
            dtype=str(tensor_manifest["dtype"]),
            payload=np.asarray(archive[tensor_manifest["payload"]], dtype=np.uint8).tobytes(),
            metadata=metadata,
            logical_nbytes=int(tensor_manifest["logical_nbytes"]),
        )
        segments.append(
            CompressedSegment(
                indices=np.asarray(archive[segment_manifest["indices"]]),
                tensor=tensor,
                precision_tier=segment_manifest["precision_tier"],
            )
        )
    return SelectiveCompressedTensor(
        shape=tuple(int(dim) for dim in manifest["shape"]),
        token_axis=int(manifest["token_axis"]),
        segments=segments,
    )


def _serialize_metadata(
    metadata: dict[str, object], prefix: str
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    manifest: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {}
    for key, value in metadata.items():
        array_key = f"{prefix}_{key}"
        if isinstance(value, np.ndarray):
            arrays[array_key] = value
            manifest[key] = {"kind": "array", "key": array_key}
        elif isinstance(value, np.generic):
            manifest[key] = {"kind": "scalar", "value": value.item()}
        elif isinstance(value, (str, int, float, bool)) or value is None:
            manifest[key] = {"kind": "scalar", "value": value}
        elif isinstance(value, (bytes, bytearray)):
            arrays[array_key] = np.frombuffer(bytes(value), dtype=np.uint8).copy()
            manifest[key] = {"kind": "bytes", "key": array_key}
        else:
            raise TypeError(f"unsupported KV page metadata value for {key!r}: {type(value).__name__}")
    return manifest, arrays


def _deserialize_metadata(
    manifest: dict[str, object], archive: np.lib.npyio.NpzFile
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key, raw_entry in manifest.items():
        entry = dict(raw_entry)
        kind = entry["kind"]
        if kind == "array":
            metadata[key] = np.asarray(archive[entry["key"]])
        elif kind == "scalar":
            metadata[key] = entry["value"]
        elif kind == "bytes":
            metadata[key] = np.asarray(archive[entry["key"]], dtype=np.uint8).tobytes()
        else:
            raise ValueError(f"unsupported KV page metadata kind for {key!r}: {kind!r}")
    return metadata
