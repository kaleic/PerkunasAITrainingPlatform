from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from perkunas_training.model.configuration import PerkunasConfig


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: PerkunasConfig,
    step: int,
    metadata: dict[str, Any],
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    step_dir = checkpoint_dir / f"step_{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    torch.save(module.state_dict(), step_dir / "model.pt")
    torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), step_dir / "scaler.pt")
    config.save_json(step_dir / "config.json")
    state = {"step": step, "metadata": metadata}
    (step_dir / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    (step_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    update_latest(checkpoint_dir, step_dir)
    return step_dir


def update_latest(checkpoint_dir: Path, step_dir: Path) -> None:
    latest = checkpoint_dir / "latest"
    checkpoint_root = checkpoint_dir.resolve()
    latest_resolved = latest.resolve() if latest.exists() else latest.absolute()
    if not str(latest_resolved).lower().startswith(str(checkpoint_root).lower()):
        raise RuntimeError("refusing to replace latest checkpoint outside checkpoint root")
    tmp = checkpoint_dir / "latest.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(step_dir, tmp)
    if latest.exists():
        shutil.rmtree(latest)
    os.replace(tmp, latest)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not (checkpoint_path / "_SUCCESS").exists():
        raise FileNotFoundError(f"checkpoint is incomplete: {checkpoint_path}")
    module = model.module if hasattr(model, "module") else model
    state = torch.load(checkpoint_path / "model.pt", map_location=map_location)
    module.load_state_dict(state)
    if optimizer is not None and (checkpoint_path / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt", map_location=map_location))
    if scaler is not None and (checkpoint_path / "scaler.pt").exists():
        scaler.load_state_dict(torch.load(checkpoint_path / "scaler.pt", map_location=map_location))
    return json.loads((checkpoint_path / "trainer_state.json").read_text(encoding="utf-8"))
