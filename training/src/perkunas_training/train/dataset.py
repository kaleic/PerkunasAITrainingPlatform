from __future__ import annotations

from bisect import bisect_right
from glob import glob
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class PackedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, shard_glob: str):
        self.paths = sorted(glob(shard_glob))
        if not self.paths:
            raise FileNotFoundError(f"no token shards matched {shard_glob}")
        self.arrays = [np.load(path, mmap_mode="r") for path in self.paths]
        self.cumulative: list[int] = []
        total = 0
        for array in self.arrays:
            if array.ndim != 2 or array.shape[1] < 2:
                raise ValueError("token shard arrays must have shape [blocks, sequence_length + 1]")
            total += int(array.shape[0])
            self.cumulative.append(total)
        self.sequence_length = int(self.arrays[0].shape[1] - 1)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        array_index = bisect_right(self.cumulative, index)
        previous = 0 if array_index == 0 else self.cumulative[array_index - 1]
        row = self.arrays[array_index][index - previous]
        tokens = torch.from_numpy(np.asarray(row, dtype=np.int64))
        return tokens[:-1], tokens[1:]


class LocalityPreservingPackedTokenDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Shuffle shard order while reading rows sequentially inside each shard.

    PyTorch's map-style RandomSampler builds a full index permutation and turns
    each batch into random mmap reads across many files. This iterable keeps the
    useful randomness at file granularity without destroying locality.
    """

    def __init__(self, shard_glob: str, *, seed: int = 0, shuffle_shards: bool = True):
        self.paths = [Path(path) for path in sorted(glob(shard_glob))]
        if not self.paths:
            raise FileNotFoundError(f"no token shards matched {shard_glob}")
        first = np.load(self.paths[0], mmap_mode="r")
        if first.ndim != 2 or first.shape[1] < 2:
            raise ValueError("token shard arrays must have shape [blocks, sequence_length + 1]")
        self.sequence_length = int(first.shape[1] - 1)
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self._epoch = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        epoch = self._epoch
        self._epoch += 1

        paths = list(self.paths)
        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed + epoch)
            rng.shuffle(paths)
        if worker is not None:
            paths = paths[worker.id :: worker.num_workers]

        for path in paths:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2 or int(array.shape[1] - 1) != self.sequence_length:
                raise ValueError(
                    f"token shard {path} has shape {array.shape}; expected sequence length "
                    f"{self.sequence_length + 1}"
                )
            for row_index in range(array.shape[0]):
                tokens = torch.from_numpy(np.asarray(array[row_index], dtype=np.int64))
                yield tokens[:-1], tokens[1:]


def dataset_summary(shard_glob: str) -> dict[str, int | str]:
    dataset = PackedTokenDataset(shard_glob)
    return {
        "shards": len(dataset.paths),
        "blocks": len(dataset),
        "sequence_length": dataset.sequence_length,
        "tokens": len(dataset) * (dataset.sequence_length + 1),
        "glob": shard_glob,
    }
