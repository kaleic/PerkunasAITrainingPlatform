from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
from typing import Any

from perkunas_training.utils.io import read_json, write_json


def build_manifest(stage_dirs: list[str | Path], output: str | Path) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    for stage_dir in stage_dirs:
        for path in sorted(glob(str(Path(stage_dir) / "manifest.json"))):
            manifests.append({"path": path, "content": read_json(path)})
    result = {"complete": True, "manifests": manifests}
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an aggregate Perkunas data manifest")
    parser.add_argument(
        "--stage-dir",
        action="append",
        default=["training/data/prepared", "training/data/dedup", "training/data/tokenized"],
    )
    parser.add_argument("--output", default="training/reports/data_manifest.json")
    args = parser.parse_args()
    result = build_manifest(args.stage_dir, args.output)
    print(f"Wrote {args.output} with {len(result['manifests'])} stage manifests")


if __name__ == "__main__":
    main()
