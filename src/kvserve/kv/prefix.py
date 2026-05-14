from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PrefixHandle:
    tenant_id: str
    model_id: str
    prefix_hash: str
    simhash: int
    token_count: int
    kv_ref: str
    created_at: float
    last_used_at: float
    hits: int = 0


class PrefixKVIndex:
    def __init__(self, near_match_hamming: int = 3):
        self.near_match_hamming = near_match_hamming
        self._by_exact: dict[tuple[str, str, str], PrefixHandle] = {}
        self._by_tenant_model: dict[tuple[str, str], list[PrefixHandle]] = defaultdict(list)
        self.lookup_count = 0
        self.hit_count = 0

    def lookup(
        self,
        tenant_id: str,
        model_id: str,
        prefix_text: str,
        allow_near_match: bool = True,
    ) -> PrefixHandle | None:
        self.lookup_count += 1
        prefix_hash = stable_prefix_hash(prefix_text)
        exact_key = (tenant_id, model_id, prefix_hash)
        exact = self._by_exact.get(exact_key)
        if exact is not None:
            exact.hits += 1
            self.hit_count += 1
            exact.last_used_at = time.time()
            return exact

        if not allow_near_match:
            return None

        target_simhash = simhash_text(prefix_text)
        best: PrefixHandle | None = None
        best_distance = 65
        for candidate in self._by_tenant_model.get((tenant_id, model_id), []):
            distance = hamming_distance(target_simhash, candidate.simhash)
            if distance < best_distance:
                best = candidate
                best_distance = distance
        if best is not None and best_distance <= self.near_match_hamming:
            best.hits += 1
            self.hit_count += 1
            best.last_used_at = time.time()
            return best
        return None

    def insert(self, tenant_id: str, model_id: str, prefix_text: str, kv_ref: str) -> PrefixHandle:
        now = time.time()
        prefix_hash = stable_prefix_hash(prefix_text)
        handle = PrefixHandle(
            tenant_id=tenant_id,
            model_id=model_id,
            prefix_hash=prefix_hash,
            simhash=simhash_text(prefix_text),
            token_count=len(tokenize_for_prefix(prefix_text)),
            kv_ref=kv_ref,
            created_at=now,
            last_used_at=now,
        )
        key = (tenant_id, model_id, prefix_hash)
        previous = self._by_exact.get(key)
        if previous is not None:
            previous.kv_ref = kv_ref
            previous.last_used_at = now
            return previous
        self._by_exact[key] = handle
        self._by_tenant_model[(tenant_id, model_id)].append(handle)
        return handle

    def evict_tenant(self, tenant_id: str) -> int:
        removed = 0
        for key in list(self._by_exact):
            if key[0] == tenant_id:
                del self._by_exact[key]
                removed += 1
        for key in list(self._by_tenant_model):
            if key[0] == tenant_id:
                del self._by_tenant_model[key]
        return removed

    def stats(self) -> dict[str, Any]:
        entries = len(self._by_exact)
        return {"entries": entries, "hits": self.hit_count, "lookups": self.lookup_count}


def stable_prefix_hash(text: str) -> str:
    normalized = normalize_prefix(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_prefix(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def tokenize_for_prefix(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", normalize_prefix(text), flags=re.UNICODE)


def simhash_text(text: str) -> int:
    weights = [0] * 64
    for token in tokenize_for_prefix(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())
