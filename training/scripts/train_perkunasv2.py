from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SRC = PROJECT_ROOT / "training" / "src"
sys.path.insert(0, str(TRAINING_SRC))

from perkunas_training.perkunasv2.train_perkunasv2 import main  # noqa: E402


if __name__ == "__main__":
    main()
