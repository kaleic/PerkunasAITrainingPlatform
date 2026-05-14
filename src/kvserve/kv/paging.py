from __future__ import annotations

import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from kvserve.kv.compression import SelectiveCompressedTensor


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
        path = self.nvme_dir / f"{page.page_id}.pkl"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(page.payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        page.path = path

    def _read_from_nvme(self, page: KVPage) -> SelectiveCompressedTensor:
        if page.path is None:
            raise RuntimeError(f"page {page.page_id} has no NVMe path")
        with page.path.open("rb") as fh:
            return pickle.load(fh)
