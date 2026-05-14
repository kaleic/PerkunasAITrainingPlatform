import glob
import numpy as np

train_files = glob.glob("training/data/perkunas_pilot/tokenized/train_*.npy")
val_files = glob.glob("training/data/perkunas_pilot/tokenized/val_*.npy")

train_tokens = 0
val_tokens = 0

for f in train_files:
    arr = np.load(f, mmap_mode="r")
    train_tokens += arr.shape[0]

for f in val_files:
    arr = np.load(f, mmap_mode="r")
    val_tokens += arr.shape[0]

print(f"Train shards: {len(train_files)}")
print(f"Val shards: {len(val_files)}")
print(f"Train tokens: {train_tokens:,}")
print(f"Val tokens: {val_tokens:,}")
print(f"Total tokens: {train_tokens + val_tokens:,}")
