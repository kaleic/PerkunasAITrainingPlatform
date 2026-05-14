from __future__ import annotations

import hashlib
import re


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def canonical_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(canonical_for_hash(text).encode("utf-8", "replace")).hexdigest()


def tokenize_for_hash(text: str, max_tokens: int | None = None) -> list[str]:
    normalized = canonical_for_hash(text)
    if max_tokens is None:
        return TOKEN_RE.findall(normalized)
    if max_tokens <= 0:
        return []
    tokens: list[str] = []
    head = max_tokens // 2
    tail = max_tokens - head
    for match in TOKEN_RE.finditer(normalized):
        tokens.append(match.group(0))
        if len(tokens) >= max_tokens:
            break
    if len(tokens) < max_tokens:
        return tokens
    tail_tokens = [match.group(0) for match in TOKEN_RE.finditer(normalized[-200_000:])][-tail:]
    return tokens[:head] + tail_tokens


def simhash64(text: str, max_tokens: int | None = None) -> int:
    weights = [0] * 64
    for token in tokenize_for_hash(text, max_tokens=max_tokens):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming64(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def deterministic_split(value: str, validation_fraction: float) -> str:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    score = int.from_bytes(digest, "big") / float(2**64 - 1)
    return "val" if score < validation_fraction else "train"
