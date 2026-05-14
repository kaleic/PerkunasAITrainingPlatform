from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PruneResult:
    key: np.ndarray
    value: np.ndarray
    retained_indices: np.ndarray
    pruned_tokens: int


def prune_kv_tokens(
    key: np.ndarray,
    value: np.ndarray,
    prune_fraction: float,
    attention_scores: np.ndarray | None,
    recent_window: int,
    token_axis: int = -2,
) -> PruneResult:
    if prune_fraction <= 0.0:
        token_axis = token_axis if token_axis >= 0 else key.ndim + token_axis
        token_count = key.shape[token_axis]
        return PruneResult(
            key=key,
            value=value,
            retained_indices=np.arange(token_count, dtype=np.int32),
            pruned_tokens=0,
        )
    if key.shape != value.shape:
        raise ValueError("key and value tensors must have identical shape for pruning")
    token_axis = token_axis if token_axis >= 0 else key.ndim + token_axis
    token_count = key.shape[token_axis]
    if token_count <= recent_window:
        return PruneResult(
            key=key,
            value=value,
            retained_indices=np.arange(token_count, dtype=np.int32),
            pruned_tokens=0,
        )

    prune_count = int(token_count * min(max(prune_fraction, 0.0), 0.80))
    prune_count = min(prune_count, token_count - recent_window)
    if prune_count <= 0:
        return PruneResult(
            key=key,
            value=value,
            retained_indices=np.arange(token_count, dtype=np.int32),
            pruned_tokens=0,
        )

    candidates = np.arange(0, token_count - recent_window, dtype=np.int32)
    if attention_scores is None:
        scores = kv_energy_scores(key, value, token_axis)
    else:
        scores = np.asarray(attention_scores, dtype=np.float32).reshape(token_count)
    candidate_scores = scores[candidates]
    prune_positions = np.argpartition(candidate_scores, prune_count - 1)[:prune_count]
    pruned = set(candidates[prune_positions].tolist())
    retained = np.array([idx for idx in range(token_count) if idx not in pruned], dtype=np.int32)
    retained.sort()
    pruned_key = np.take(key, retained, axis=token_axis)
    pruned_value = np.take(value, retained, axis=token_axis)
    return PruneResult(
        key=pruned_key,
        value=pruned_value,
        retained_indices=retained,
        pruned_tokens=token_count - retained.size,
    )


def kv_energy_scores(key: np.ndarray, value: np.ndarray, token_axis: int) -> np.ndarray:
    key_moved = np.moveaxis(np.asarray(key, dtype=np.float32), token_axis, 0)
    value_moved = np.moveaxis(np.asarray(value, dtype=np.float32), token_axis, 0)
    key_energy = np.mean(np.abs(key_moved), axis=tuple(range(1, key_moved.ndim)))
    value_energy = np.mean(np.abs(value_moved), axis=tuple(range(1, value_moved.ndim)))
    scores = key_energy + value_energy
    if np.max(scores) > np.min(scores):
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    return scores.astype(np.float32)
