from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_SRC = ROOT / "training" / "src"
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))


if __name__ == "__main__":
    from perkunas_training.perkunasv2.c4_training import main

    main()
