from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp = Path(fh.name)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    count = 0
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
        tmp = Path(fh.name)
    os.replace(tmp, path)
    return count


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
