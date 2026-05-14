from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from perkunas_training.config import DataConfig, DatasetSourceConfig
from perkunas_training.data.inspect import NumericSketch
from perkunas_training.data.sources import (
    choose_mapping_text_field,
    load_hf_dataset,
    select_metadata,
)
from perkunas_training.utils.hashing import sha256_text
from perkunas_training.utils.io import ensure_dir, write_json
from perkunas_training.utils.text import CONTROL_CHARS, normalize_text, word_count


def inspect_hf_dataset_source(
    source: DatasetSourceConfig,
    reports_dir: str | Path,
    *,
    max_samples: int = 1000,
) -> dict[str, Any]:
    if source.type != "hf_dataset":
        raise ValueError("inspect_hf_dataset_source expects a source with type='hf_dataset'")
    reports_dir = ensure_dir(reports_dir)
    dataset = load_hf_dataset(source)
    if isinstance(dataset, Mapping):
        split_name = source.split or "train"
        if split_name not in dataset:
            available = ", ".join(str(key) for key in dataset.keys())
            raise ValueError(
                f"split {split_name!r} not found in {source.dataset_name!r}; "
                f"available splits: {available}"
            )
        dataset_iterable = dataset[split_name]
    else:
        dataset_iterable = dataset

    features = getattr(dataset_iterable, "features", None)
    feature_schema = (
        {key: str(value) for key, value in features.items()} if hasattr(features, "items") else None
    )
    nulls = Counter()
    language = Counter()
    license_counts = Counter()
    collection = Counter()
    source_counts = Counter()
    duplicate_hashes: set[str] = set()
    duplicate_count = 0
    empty_rows = 0
    short_rows = 0
    control_char_rows = 0
    length_sketch = NumericSketch()
    word_sketch = NumericSketch()
    text_field = source.text_field
    metadata_fields: set[str] = set()
    sampled_rows = 0

    for sampled_rows, row in enumerate(dataset_iterable, start=1):
        if sampled_rows > max_samples:
            sampled_rows -= 1
            break
        if not isinstance(row, Mapping):
            row = {"text": row}
        if text_field is None:
            text_field = choose_mapping_text_field(row, source.text_field)
        for key, value in row.items():
            if value is None:
                nulls[key] += 1
        raw_text = row.get(text_field)
        if raw_text is None:
            empty_rows += 1
            continue
        text = normalize_text(str(raw_text))
        if not text:
            empty_rows += 1
            continue
        if CONTROL_CHARS.search(text):
            control_char_rows += 1
        if len(text) < 200:
            short_rows += 1
        length_sketch.add(len(text))
        word_sketch.add(word_count(text))
        doc_hash = sha256_text(text)
        if doc_hash in duplicate_hashes:
            duplicate_count += 1
        duplicate_hashes.add(doc_hash)
        metadata = select_metadata(row, text_field, source.metadata_fields)
        metadata_fields.update(metadata)
        for field_name, counter in (
            ("language", language),
            ("license", license_counts),
            ("collection", collection),
            ("source", source_counts),
        ):
            value = metadata.get(field_name)
            if value is not None:
                counter[str(value)] += 1

    profile = {
        "source": asdict(source),
        "dataset_name": source.dataset_name,
        "dataset_config": source.dataset_config,
        "split": source.split,
        "streaming": source.streaming,
        "max_samples": max_samples,
        "sampled_rows": sampled_rows,
        "features": feature_schema,
        "selected_text_field": text_field,
        "metadata_fields": sorted(metadata_fields),
        "nulls": dict(nulls),
        "empty_text_rows": empty_rows,
        "short_text_lt_200_chars": short_rows,
        "exact_duplicate_rows": duplicate_count,
        "unique_text_hashes": len(duplicate_hashes),
        "control_char_rows": control_char_rows,
        "language_top": language.most_common(25),
        "license_top": license_counts.most_common(25),
        "collection_top": collection.most_common(25),
        "source_top": source_counts.most_common(25),
        "char_length": length_sketch.stats(),
        "word_count": word_sketch.stats(),
    }
    write_json(reports_dir / "hf_common_corpus_profile.json", profile)
    (reports_dir / "hf_common_corpus_profile.md").write_text(
        render_hf_markdown_report(profile), encoding="utf-8"
    )
    return profile


def render_hf_markdown_report(profile: dict[str, Any]) -> str:
    lines = [
        "# Perkunas Hugging Face Dataset Inspection Report",
        "",
        f"Dataset: `{profile['dataset_name']}`",
        f"Split: `{profile['split']}`",
        f"Streaming: `{profile['streaming']}`",
        f"Sampled rows: `{profile['sampled_rows']}`",
        f"Selected text field: `{profile['selected_text_field']}`",
        f"Metadata fields: `{', '.join(profile['metadata_fields'])}`",
        "",
        "## Quality Signals",
        "",
        f"- Empty text rows: `{profile['empty_text_rows']}`",
        f"- Short rows (<200 chars): `{profile['short_text_lt_200_chars']}`",
        f"- Exact duplicate rows in sample: `{profile['exact_duplicate_rows']}`",
        f"- Unique text hashes in sample: `{profile['unique_text_hashes']}`",
        f"- Control-character rows: `{profile['control_char_rows']}`",
        "",
        "## Length Distributions",
        "",
        f"- Characters: `{profile['char_length']}`",
        f"- Words: `{profile['word_count']}`",
        "",
        "## Top Languages",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in profile["language_top"][:15])
    lines.extend(["", "## Top Licenses", ""])
    lines.extend(f"- `{name}`: {count}" for name, count in profile["license_top"][:15])
    lines.extend(["", "## Top Collections", ""])
    lines.extend(f"- `{name}`: {count}" for name, count in profile["collection_top"][:15])
    if profile.get("features"):
        lines.extend(["", "## Features", "", "| Column | Type |", "| --- | --- |"])
        lines.extend(f"| `{key}` | `{value}` |" for key, value in profile["features"].items())
    return "\n".join(lines) + "\n"


def first_hf_source(config: DataConfig) -> DatasetSourceConfig:
    for source in [*config.datasets, *config.validation_datasets]:
        if source.type == "hf_dataset":
            return source
    raise ValueError("config does not contain an hf_dataset source")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Hugging Face dataset source")
    parser.add_argument("--config", default="training/configs/data_hf_common_corpus.yaml")
    parser.add_argument("--dataset-name", help="Override Hugging Face dataset name")
    parser.add_argument("--dataset-config", help="Override Hugging Face dataset config/subset")
    parser.add_argument("--split", help="Override split")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache-dir")
    parser.add_argument("--reports-dir")
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    config = DataConfig.from_yaml(args.config)
    source = first_hf_source(config)
    if args.dataset_name:
        source.dataset_name = args.dataset_name
    if args.dataset_config:
        source.dataset_config = args.dataset_config
    if args.split:
        source.split = args.split
    if args.streaming is not None:
        source.streaming = args.streaming
    if args.cache_dir:
        source.cache_dir = args.cache_dir
    reports_dir = args.reports_dir or config.reports_dir
    profile = inspect_hf_dataset_source(source, reports_dir, max_samples=args.max_samples)
    print(f"Wrote {Path(reports_dir) / 'hf_common_corpus_profile.md'}")
    print(f"Wrote {Path(reports_dir) / 'hf_common_corpus_profile.json'}")
    print(f"Selected text field: {profile['selected_text_field']}")
    print(f"Sampled rows: {profile['sampled_rows']}")


if __name__ == "__main__":
    main()
