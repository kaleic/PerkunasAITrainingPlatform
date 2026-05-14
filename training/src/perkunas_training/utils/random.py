from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        return
