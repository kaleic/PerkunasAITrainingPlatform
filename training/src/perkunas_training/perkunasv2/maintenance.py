from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from perkunas_training.perkunasv2.configuration import PerkunasV2Config
from perkunas_training.perkunasv2.shard_store import build_module_for_shard, shard_names


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def audit_run(run_dir: str | Path, *, include_optimizer: bool = True) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = PerkunasV2Config.from_json(run_dir / "config.json")
    expected_shards = shard_names(config)
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checked_unix": time.time(),
        "config": config.to_dict(),
        "param_shards": [],
        "optimizer_shards": [],
        "issues": [],
        "summary": {
            "param_tensors": 0,
            "optimizer_tensors": 0,
            "nonfinite_param_tensors": 0,
            "nonfinite_optimizer_tensors": 0,
            "missing_param_shards": 0,
            "missing_optimizer_shards": 0,
            "shape_mismatches": 0,
            "optimizer_step_min": None,
            "optimizer_step_max": None,
        },
    }

    optimizer_steps: list[int] = []
    for shard_name in expected_shards:
        param_path = run_dir / "shards" / "params" / f"{shard_name}.pt"
        if not param_path.exists():
            report["summary"]["missing_param_shards"] += 1
            report["issues"].append({"severity": "error", "message": f"missing {param_path}"})
            continue
        payload = torch.load(param_path, map_location="cpu")
        expected_shapes = module_state_shapes(config, shard_name)
        shard_report = audit_state_dict(
            shard_name,
            payload.get("state_dict", {}),
            expected_shapes=expected_shapes,
        )
        report["param_shards"].append(shard_report)
        report["summary"]["param_tensors"] += shard_report["tensor_count"]
        report["summary"]["nonfinite_param_tensors"] += shard_report["nonfinite_tensor_count"]
        report["summary"]["shape_mismatches"] += shard_report["shape_mismatch_count"]
        report["issues"].extend(shard_report["issues"])

        if include_optimizer:
            optim_path = run_dir / "shards" / "optim" / f"{shard_name}.pt"
            if not optim_path.exists():
                report["summary"]["missing_optimizer_shards"] += 1
                report["issues"].append(
                    {"severity": "warning", "message": f"missing optimizer {optim_path}"}
                )
                continue
            optim_payload = torch.load(optim_path, map_location="cpu")
            optim_report = audit_optimizer_state(shard_name, optim_payload.get("state", {}))
            report["optimizer_shards"].append(optim_report)
            report["summary"]["optimizer_tensors"] += optim_report["tensor_count"]
            report["summary"]["nonfinite_optimizer_tensors"] += optim_report[
                "nonfinite_tensor_count"
            ]
            report["issues"].extend(optim_report["issues"])
            optimizer_steps.extend(optim_report["steps"])

    if optimizer_steps:
        report["summary"]["optimizer_step_min"] = min(optimizer_steps)
        report["summary"]["optimizer_step_max"] = max(optimizer_steps)
        if min(optimizer_steps) != max(optimizer_steps):
            report["issues"].append(
                {
                    "severity": "warning",
                    "message": (
                        "optimizer state steps are not uniform: "
                        f"min={min(optimizer_steps)} max={max(optimizer_steps)}"
                    ),
                }
            )
    report["summary"]["issue_count"] = len(report["issues"])
    return report


def module_state_shapes(config: PerkunasV2Config, shard_name: str) -> dict[str, tuple[int, ...]]:
    module = build_module_for_shard(config, shard_name)
    return {name: tuple(tensor.shape) for name, tensor in module.state_dict().items()}


def audit_state_dict(
    shard_name: str,
    state_dict: dict[str, Any],
    *,
    expected_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    nonfinite_count = 0
    shape_mismatches = 0
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            issues.append(
                {
                    "severity": "error",
                    "message": f"{shard_name}.{name} is not a tensor",
                }
            )
            continue
        row = tensor_stats(name, tensor)
        expected_shape = expected_shapes.get(name)
        if expected_shape is None:
            issues.append(
                {
                    "severity": "warning",
                    "message": f"{shard_name}.{name} is unexpected in state_dict",
                }
            )
        elif tuple(tensor.shape) != expected_shape:
            shape_mismatches += 1
            issues.append(
                {
                    "severity": "error",
                    "message": (
                        f"{shard_name}.{name} shape mismatch: "
                        f"actual={tuple(tensor.shape)} expected={expected_shape}"
                    ),
                }
            )
        if row["nan_count"] or row["inf_count"]:
            nonfinite_count += 1
            issues.append(
                {
                    "severity": "error",
                    "message": (
                        f"{shard_name}.{name} has nan={row['nan_count']} "
                        f"inf={row['inf_count']}"
                    ),
                }
            )
        rows.append(row)

    for missing_name in sorted(set(expected_shapes) - set(state_dict)):
        shape_mismatches += 1
        issues.append(
            {
                "severity": "error",
                "message": f"{shard_name}.{missing_name} is missing from state_dict",
            }
        )
    return {
        "shard_name": shard_name,
        "tensor_count": len(rows),
        "nonfinite_tensor_count": nonfinite_count,
        "shape_mismatch_count": shape_mismatches,
        "tensors": rows,
        "issues": issues,
    }


def audit_optimizer_state(shard_name: str, state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    steps: list[int] = []
    nonfinite_count = 0
    for param_name, item in state.items():
        step = int(item.get("step", 0))
        steps.append(step)
        for state_name, value in item.items():
            if state_name == "step":
                continue
            if not torch.is_tensor(value):
                continue
            row = tensor_stats(f"{param_name}.{state_name}", value)
            if row["nan_count"] or row["inf_count"]:
                nonfinite_count += 1
                issues.append(
                    {
                        "severity": "error",
                        "message": (
                            f"{shard_name}.{param_name}.{state_name} has "
                            f"nan={row['nan_count']} inf={row['inf_count']}"
                        ),
                    }
                )
            rows.append(row)
    return {
        "shard_name": shard_name,
        "tensor_count": len(rows),
        "nonfinite_tensor_count": nonfinite_count,
        "steps": sorted(set(steps)),
        "tensors": rows,
        "issues": issues,
    }


def tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    data = tensor.detach()
    finite = torch.isfinite(data)
    nan_count = int(torch.isnan(data).sum().item())
    inf_count = int(torch.isinf(data).sum().item())
    row: dict[str, Any] = {
        "name": name,
        "dtype": str(data.dtype).replace("torch.", ""),
        "shape": list(data.shape),
        "numel": int(data.numel()),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    if finite.any():
        values = data[finite].float()
        row.update(
            {
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
                "rms": float(values.square().mean().sqrt().item()),
                "absmax": float(values.abs().max().item()),
            }
        )
    else:
        row.update({"min": None, "max": None, "mean": None, "std": None, "rms": None, "absmax": None})
    return row


def prepare_backup(
    source_run_dir: str | Path,
    backup_run_dir: str | Path,
    *,
    tokenizer_dir: str | Path | None = None,
    sanitize_nonfinite: bool = True,
) -> dict[str, Any]:
    source = Path(source_run_dir).resolve()
    backup = Path(backup_run_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    if source == backup:
        raise ValueError("backup directory must differ from source directory")

    shutil.copytree(source, backup, ignore=shutil.ignore_patterns("transactions"))
    report: dict[str, Any] = {
        "source_run_dir": str(source),
        "backup_run_dir": str(backup),
        "created_unix": time.time(),
        "tokenizer_copied": False,
        "metadata_paths_rewritten": False,
        "sanitized_files": [],
    }
    rewrite_metadata_paths(backup)
    report["metadata_paths_rewritten"] = True

    if tokenizer_dir is not None:
        copied = copy_tokenizer(Path(tokenizer_dir), backup / "tokenizer")
        report["tokenizer_copied"] = bool(copied)
        report["tokenizer_files"] = copied

    if sanitize_nonfinite:
        report["sanitized_files"] = sanitize_backup_nonfinite(backup)

    audit = audit_run(backup)
    report["audit_summary"] = audit["summary"]
    (backup / "weight_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (backup / "backup_preparation_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def rewrite_metadata_paths(run_dir: Path) -> None:
    metadata_path = run_dir / "shards" / "metadata.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for shard in metadata.get("shards", []):
        name = shard["name"]
        shard["params_path"] = str(run_dir / "shards" / "params" / f"{name}.pt")
        shard["optim_path"] = str(run_dir / "shards" / "optim" / f"{name}.pt")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def copy_tokenizer(tokenizer_dir: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in TOKENIZER_FILES:
        source = tokenizer_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
            copied.append(name)
    return copied


def sanitize_backup_nonfinite(run_dir: Path) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for kind in ("params", "optim"):
        root = run_dir / "shards" / kind
        for path in sorted(root.glob("*.pt")):
            payload = torch.load(path, map_location="cpu")
            replacements = sanitize_payload_nonfinite(payload)
            if replacements:
                torch.save(payload, path)
                changed.append(
                    {
                        "path": str(path),
                        "kind": kind,
                        "replacements": replacements,
                    }
                )
    return changed


def sanitize_payload_nonfinite(payload: Any, prefix: str = "") -> list[dict[str, Any]]:
    replacements: list[dict[str, Any]] = []
    if torch.is_tensor(payload):
        nonfinite_mask = ~torch.isfinite(payload)
        count = int(nonfinite_mask.sum().item())
        if count:
            payload.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            replacements.append({"tensor": prefix, "replaced_nonfinite": count})
        return replacements
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            replacements.extend(sanitize_payload_nonfinite(value, child_prefix))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            replacements.extend(sanitize_payload_nonfinite(value, f"{prefix}[{index}]"))
    elif isinstance(payload, tuple):
        for index, value in enumerate(payload):
            replacements.extend(sanitize_payload_nonfinite(value, f"{prefix}[{index}]"))
    return replacements


def print_audit_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        json.dumps(
            {
                "run_dir": report["run_dir"],
                "param_tensors": summary["param_tensors"],
                "optimizer_tensors": summary["optimizer_tensors"],
                "nonfinite_param_tensors": summary["nonfinite_param_tensors"],
                "nonfinite_optimizer_tensors": summary["nonfinite_optimizer_tensors"],
                "shape_mismatches": summary["shape_mismatches"],
                "optimizer_step_min": summary["optimizer_step_min"],
                "optimizer_step_max": summary["optimizer_step_max"],
                "issue_count": summary["issue_count"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit or prepare Perkunasv2 shard-native runs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--run-dir", required=True)
    audit_parser.add_argument("--output")
    audit_parser.add_argument("--no-optimizer", action="store_true")

    backup_parser = subparsers.add_parser("backup")
    backup_parser.add_argument("--source-run-dir", required=True)
    backup_parser.add_argument("--backup-run-dir", required=True)
    backup_parser.add_argument("--tokenizer-dir")
    backup_parser.add_argument("--no-sanitize", action="store_true")

    args = parser.parse_args()
    if args.command == "audit":
        report = audit_run(args.run_dir, include_optimizer=not args.no_optimizer)
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print_audit_summary(report)
        return

    if args.command == "backup":
        report = prepare_backup(
            args.source_run_dir,
            args.backup_run_dir,
            tokenizer_dir=args.tokenizer_dir,
            sanitize_nonfinite=not args.no_sanitize,
        )
        print(json.dumps(report, indent=2))
        return

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
