import glob
import numpy as np
import math

def count_tokens(files):
    total = 0
    for f in files:
        arr = np.load(f, mmap_mode="r")
        total += math.prod(arr.shape)
    return total

train_files = glob.glob("training/data/perkunas_pilot/tokenized/train_*.npy")
val_files = glob.glob("training/data/perkunas_pilot/tokenized/val_*.npy")

train_tokens = count_tokens(train_files)
val_tokens = count_tokens(val_files)

print(f"Train shards: {len(train_files)}")
print(f"Val shards: {len(val_files)}")
print(f"Train tokens: {train_tokens:,}")
print(f"Val tokens: {val_tokens:,}")
print(f"Total tokens: {train_tokens + val_tokens:,}")
