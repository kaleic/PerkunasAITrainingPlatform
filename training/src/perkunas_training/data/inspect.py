from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from perkunas_training.config import DataConfig
from perkunas_training.utils.hashing import sha256_text
from perkunas_training.utils.io import ensure_dir, write_json
from perkunas_training.utils.text import CONTROL_CHARS, normalize_text, word_count


TEXT_NAME_HINTS = ("text", "content", "body", "document", "raw", "article")
METADATA_HINTS = (
    "identifier",
    "id",
    "source",
    "url",
    "language",
    "license",
    "date",
    "collection",
    "title",
    "creator",
    "curator",
    "token_count",
    "word_count",
)


@dataclass(slots=True)
class NumericSketch:
    values: list[float] = field(default_factory=list)

    def add(self, value: Any) -> None:
        if value is None:
            return
        try:
            self.values.append(float(value))
        except (TypeError, ValueError):
            return

    def stats(self) -> dict[str, float | int | None]:
        if not self.values:
            return {"count": 0}
        arr = np.asarray(self.values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "min": float(arr.min()),
            "p01": float(np.quantile(arr, 0.01)),
            "p05": float(np.quantile(arr, 0.05)),
            "p50": float(np.quantile(arr, 0.50)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }


def inspect_parquet_files(
    paths: list[str | Path],
    reports_dir: str | Path,
    *,
    batch_size: int = 1024,
    configured_text_field: str | None = None,
    max_profile_rows_per_file: int | None = None,
) -> dict[str, Any]:
    reports_dir = ensure_dir(reports_dir)
    path_objects = [Path(path) for path in paths]
    if not path_objects:
        raise ValueError("at least one parquet input path is required")

    file_reports: list[dict[str, Any]] = []
    combined_language = Counter()
    combined_license = Counter()
    combined_collection = Counter()
    combined_nulls = Counter()
    combined_rows = 0
    duplicate_hashes: set[str] = set()
    duplicate_count = 0
    length_sketch = NumericSketch()
    word_sketch = NumericSketch()
    token_sketch = NumericSketch()
    short_rows = 0
    empty_rows = 0
    control_char_rows = 0
    encoding_issue_rows = 0

    for path in path_objects:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        candidate_text_fields = infer_text_fields(schema)
        text_field = choose_text_field(schema, configured_text_field, candidate_text_fields)
        metadata_fields = infer_metadata_fields(schema)
        row_group_rows = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
        file_report = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "num_rows": pf.metadata.num_rows,
            "num_row_groups": pf.metadata.num_row_groups,
            "created_by": pf.metadata.created_by,
            "schema": schema_to_json(schema),
            "candidate_text_fields": candidate_text_fields,
            "selected_text_field": text_field,
            "metadata_fields": metadata_fields,
            "row_group_rows": row_group_rows,
        }
        file_reports.append(file_report)
        combined_rows += pf.metadata.num_rows

        columns = list(dict.fromkeys([*schema.names]))
        profiled_rows = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            batch_dict = batch.to_pydict()
            rows = batch.num_rows
            for column in schema.names:
                combined_nulls[column] += sum(value is None for value in batch_dict[column])
            for index in range(rows):
                if max_profile_rows_per_file is not None and profiled_rows >= max_profile_rows_per_file:
                    break
                raw_text = batch_dict[text_field][index]
                profiled_rows += 1
                if raw_text is None:
                    empty_rows += 1
                    continue
                try:
                    text = raw_text if isinstance(raw_text, str) else str(raw_text)
                    text.encode("utf-8")
                except UnicodeError:
                    encoding_issue_rows += 1
                    text = str(raw_text).encode("utf-8", "replace").decode("utf-8")
                if CONTROL_CHARS.search(text):
                    control_char_rows += 1
                normalized = normalize_text(text)
                if not normalized:
                    empty_rows += 1
                if len(normalized) < 200:
                    short_rows += 1
                length_sketch.add(len(normalized))
                word_sketch.add(word_count(normalized))
                if "token_count" in batch_dict:
                    token_sketch.add(batch_dict["token_count"][index])
                doc_hash = sha256_text(normalized)
                if doc_hash in duplicate_hashes:
                    duplicate_count += 1
                duplicate_hashes.add(doc_hash)
                for field_name, counter in (
                    ("language", combined_language),
                    ("license", combined_license),
                    ("collection", combined_collection),
                ):
                    if field_name in batch_dict:
                        counter[batch_dict[field_name][index]] += 1
            if max_profile_rows_per_file is not None and profiled_rows >= max_profile_rows_per_file:
                break
        file_report["profiled_rows"] = profiled_rows

    profile = {
        "input_paths": [str(path) for path in path_objects],
        "total_rows": combined_rows,
        "files": file_reports,
        "selected_text_field": file_reports[0]["selected_text_field"],
        "candidate_text_fields": file_reports[0]["candidate_text_fields"],
        "metadata_fields": sorted(set().union(*(set(f["metadata_fields"]) for f in file_reports))),
        "nulls": dict(combined_nulls),
        "empty_text_rows": empty_rows,
        "short_text_lt_200_chars": short_rows,
        "exact_duplicate_rows": duplicate_count,
        "unique_text_hashes": len(duplicate_hashes),
        "control_char_rows": control_char_rows,
        "encoding_issue_rows": encoding_issue_rows,
        "language_top": combined_language.most_common(25),
        "license_top": combined_license.most_common(25),
        "collection_top": combined_collection.most_common(25),
        "char_length": length_sketch.stats(),
        "word_count": word_sketch.stats(),
        "token_count": token_sketch.stats(),
    }
    write_json(reports_dir / "parquet_profile.json", profile)
    (reports_dir / "parquet_profile.md").write_text(render_markdown_report(profile), encoding="utf-8")
    return profile


def infer_text_fields(schema: pa.Schema) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for schema_field in schema:
        if not pa.types.is_string(schema_field.type) and not pa.types.is_large_string(schema_field.type):
            continue
        name = schema_field.name.lower()
        score = 0
        if name == "text":
            score += 100
        if any(hint in name for hint in TEXT_NAME_HINTS):
            score += 25
        if name in METADATA_HINTS:
            score -= 20
        candidates.append((score, schema_field.name))
    return [name for _, name in sorted(candidates, key=lambda item: (-item[0], item[1]))]


def choose_text_field(
    schema: pa.Schema, configured_text_field: str | None, candidates: list[str]
) -> str:
    if configured_text_field:
        if configured_text_field not in schema.names:
            raise ValueError(f"configured text field {configured_text_field!r} not found in schema")
        return configured_text_field
    if not candidates:
        raise ValueError("no string field candidates found for training text")
    return candidates[0]


def infer_metadata_fields(schema: pa.Schema) -> list[str]:
    result: list[str] = []
    for schema_field in schema:
        name = schema_field.name.lower()
        if any(hint == name or hint in name for hint in METADATA_HINTS):
            result.append(schema_field.name)
    return result


def schema_to_json(schema: pa.Schema) -> list[dict[str, str]]:
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def render_markdown_report(profile: dict[str, Any]) -> str:
    lines = [
        "# Perkunas Data Inspection Report",
        "",
        f"Total rows: `{profile['total_rows']}`",
        f"Selected text field: `{profile['selected_text_field']}`",
        f"Candidate text fields: `{', '.join(profile['candidate_text_fields'])}`",
        f"Metadata fields: `{', '.join(profile['metadata_fields'])}`",
        "",
        "## Quality Signals",
        "",
        f"- Empty text rows: `{profile['empty_text_rows']}`",
        f"- Short rows (<200 chars): `{profile['short_text_lt_200_chars']}`",
        f"- Exact duplicate rows: `{profile['exact_duplicate_rows']}`",
        f"- Unique text hashes: `{profile['unique_text_hashes']}`",
        f"- Control-character rows: `{profile['control_char_rows']}`",
        f"- Encoding issue rows: `{profile['encoding_issue_rows']}`",
        "",
        "## Length Distributions",
        "",
        f"- Characters: `{profile['char_length']}`",
        f"- Words: `{profile['word_count']}`",
        f"- Source token count: `{profile['token_count']}`",
        "",
        "## Top Languages",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in profile["language_top"][:15])
    lines.extend(["", "## Top Licenses", ""])
    lines.extend(f"- `{name}`: {count}" for name, count in profile["license_top"][:15])
    lines.extend(["", "## Top Collections", ""])
    lines.extend(f"- `{name}`: {count}" for name, count in profile["collection_top"][:15])
    lines.extend(["", "## Files", ""])
    for file_report in profile["files"]:
        lines.extend(
            [
                f"### `{file_report['path']}`",
                "",
                f"- Rows: `{file_report['num_rows']}`",
                f"- Row groups: `{file_report['num_row_groups']}`",
                f"- Size bytes: `{file_report['size_bytes']}`",
                "",
                "| Column | Type |",
                "| --- | --- |",
            ]
        )
        lines.extend(
            f"| `{field['name']}` | `{field['type']}` |" for field in file_report["schema"]
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect parquet shards for Perkunas pretraining")
    parser.add_argument("--config", default="training/configs/data.yaml")
    parser.add_argument("--input", nargs="*", help="Override parquet paths")
    parser.add_argument("--reports-dir", help="Override reports directory")
    parser.add_argument("--max-profile-rows-per-file", type=int)
    args = parser.parse_args()

    config = DataConfig.from_yaml(args.config)
    configured_paths = config.input_paths or [
        path
        for source in [*config.datasets, *config.validation_datasets]
        for path in source.paths
    ]
    paths = args.input or configured_paths
    reports_dir = args.reports_dir or config.reports_dir
    profile = inspect_parquet_files(
        paths,
        reports_dir,
        batch_size=config.batch_size,
        configured_text_field=config.text_field,
        max_profile_rows_per_file=args.max_profile_rows_per_file,
    )
    print(f"Wrote {Path(reports_dir) / 'parquet_profile.md'}")
    print(f"Wrote {Path(reports_dir) / 'parquet_profile.json'}")
    print(f"Selected text field: {profile['selected_text_field']}")


if __name__ == "__main__":
    main()
