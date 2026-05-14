import json
from pathlib import Path

import pandas as pd


INPUT_DIR = Path(r"D:\LLMProject\TrainingData\allenai\c4\validation")
OUTPUT_DIR = Path(r"D:\LLMProject\TrainingData\allenai\c4\validation_parquet")
BATCH_SIZE = 50_000


def convert_file(json_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUTPUT_DIR / f"{json_path.stem}.parquet"

    if out_path.exists():
        print(f"SKIP exists: {out_path}")
        return

    rows = []
    total = 0
    part_files = []

    print(f"Converting: {json_path.name}")

    with json_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"BAD JSON {json_path.name}:{line_num} {e}")
                continue

            rows.append({
                "text": obj.get("text", ""),
                "timestamp": obj.get("timestamp", None),
                "url": obj.get("url", None),
                "source_dataset": "allenai/c4",
                "source_split": "validation",
                "source_file": json_path.name,
                "row_index": total,
            })

            total += 1

            if len(rows) >= BATCH_SIZE:
                part_path = OUTPUT_DIR / f"{json_path.stem}.part{len(part_files):05d}.parquet"
                pd.DataFrame(rows).to_parquet(part_path, index=False)
                part_files.append(part_path)
                rows.clear()
                print(f"  wrote {total:,} rows...")

    if rows:
        part_path = OUTPUT_DIR / f"{json_path.stem}.part{len(part_files):05d}.parquet"
        pd.DataFrame(rows).to_parquet(part_path, index=False)
        part_files.append(part_path)

    if len(part_files) == 1:
        part_files[0].rename(out_path)
    else:
        dfs = [pd.read_parquet(p) for p in part_files]
        pd.concat(dfs, ignore_index=True).to_parquet(out_path, index=False)
        for p in part_files:
            p.unlink()

    print(f"DONE: {json_path.name} -> {out_path.name} rows={total:,}")


def main() -> None:
    files = sorted(INPUT_DIR.glob("c4-validation.*.json"))

    if not files:
        raise FileNotFoundError(f"No validation JSON files found in {INPUT_DIR}")

    print(f"Found {len(files)} files")

    for file in files:
        convert_file(file)


if __name__ == "__main__":
    main()