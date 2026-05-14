from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

from perkunas_training.config import DataConfig, DatasetSourceConfig
from perkunas_training.data.sources import iter_source_records
from perkunas_training.utils.hashing import sha256_text
from perkunas_training.utils.io import ensure_dir, write_json, write_jsonl
from perkunas_training.utils.text import looks_low_value, normalize_text, word_count


@dataclass(slots=True)
class NormalizeStats:
    seen: int = 0
    written: int = 0
    empty: int = 0
    too_short: int = 0
    too_long: int = 0
    too_few_words: int = 0
    language_filtered: int = 0
    license_filtered: int = 0
    collection_filtered: int = 0
    date_filtered: int = 0
    low_value: int = 0
    chunks_created: int = 0
    oversized_docs_chunked: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def normalize_corpus(config: DataConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.prepared_dir)
    manifest_path = output_dir / "manifest.json"
    if config.resume and manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete"):
            return manifest

    stats = NormalizeStats()
    shard_index = 0
    shard_rows: list[dict[str, Any]] = []
    shard_manifest: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal shard_index, shard_rows
        if not shard_rows:
            return
        shard_name = f"normalized_{shard_index:05d}.jsonl"
        shard_path = output_dir / shard_name
        count = write_jsonl(shard_path, shard_rows)
        shard_manifest.append({"path": str(shard_path), "rows": count})
        shard_index += 1
        shard_rows = []

    def add_record(record: dict[str, Any]) -> None:
        shard_rows.append(record)
        if len(shard_rows) >= config.output_shard_rows:
            flush()

    for source, source_split in iter_sources(config):
        for raw_record in iter_source_records(source, source_split, config):
            stats.seen += 1
            records = build_normalized_records(
                raw_record.text,
                row_id=raw_record.doc_id,
                source_name=raw_record.source_name,
                source_type=raw_record.source_type,
                source_split=raw_record.source_split,
                source_weight=raw_record.source_weight,
                source_path_or_dataset=raw_record.source_path_or_dataset,
                source_row=raw_record.source_row,
                metadata=raw_record.metadata,
                default_language=source.default_language,
                config=config,
                stats=stats,
            )
            for record in records:
                stats.written += 1
                add_record(record)
    flush()

    manifest = {
        "stage": "normalize",
        "complete": True,
        "input_paths": input_refs(config),
        "datasets": [asdict(source) | {"split": split} for source, split in iter_sources(config)],
        "text_field": config.text_field,
        "metadata_fields": config.metadata_fields,
        "chunking": asdict(config.chunking),
        "stats": stats.to_dict(),
        "shards": shard_manifest,
    }
    write_json(manifest_path, manifest)
    (output_dir / "normalization_report.md").write_text(render_report(manifest), encoding="utf-8")
    return manifest


def iter_sources(config: DataConfig) -> list[tuple[DatasetSourceConfig, str]]:
    if config.datasets or config.validation_datasets:
        return [(source, "train") for source in config.datasets] + [
            (source, "validation") for source in config.validation_datasets
        ]
    return [
        (
            DatasetSourceConfig(
                name="default",
                type="parquet_local",
                weight=1.0,
                paths=config.input_paths,
                text_field=config.text_field,
                metadata_fields=config.metadata_fields,
            ),
            "train",
        )
    ]


def input_refs(config: DataConfig) -> list[str]:
    refs: list[str] = []
    for source, _ in iter_sources(config):
        if source.type == "hf_dataset" and source.dataset_name:
            refs.append(source.dataset_name)
        else:
            refs.extend(source.paths)
    return refs


def build_normalized_records(
    raw_text: Any,
    *,
    row_id: str,
    source_name: str,
    source_type: str = "parquet_local",
    source_split: str,
    source_weight: float,
    source_path_or_dataset: str,
    source_row: int,
    metadata: dict[str, Any],
    default_language: str | None,
    config: DataConfig,
    stats: NormalizeStats,
) -> list[dict[str, Any]]:
    if raw_text is None:
        stats.empty += 1
        return []
    text = normalize_text(str(raw_text))
    if not text:
        stats.empty += 1
        return []
    metadata = dict(metadata)
    if default_language is not None and not metadata.get("language"):
        metadata["language"] = default_language
    language = metadata_value(metadata, "language", "lang")
    language_allowlist = config.language_allowlist or config.allowed_languages
    if language_allowlist is not None and not value_in_allowlist(language, language_allowlist):
        stats.language_filtered += 1
        return []
    license_value = metadata_value(metadata, "license", "licence", "open_type")
    license_allowlist = config.license_allowlist or config.allowed_licenses
    if license_allowlist is not None and not value_in_allowlist(license_value, license_allowlist):
        stats.license_filtered += 1
        return []
    if config.license_blocklist is not None and value_in_allowlist(
        license_value, config.license_blocklist
    ):
        stats.license_filtered += 1
        return []
    collection_value = metadata_value(metadata, "collection", "source")
    if config.collection_allowlist is not None and not value_in_allowlist(
        collection_value, config.collection_allowlist
    ):
        stats.collection_filtered += 1
        return []
    if config.collection_blocklist is not None and value_in_allowlist(
        collection_value, config.collection_blocklist
    ):
        stats.collection_filtered += 1
        return []
    date_value = metadata_value(metadata, "date", "timestamp", "created", "publication_date")
    if date_value is not None:
        date_int = coerce_year(date_value)
        if config.min_date is not None and date_int is not None and date_int < config.min_date:
            stats.date_filtered += 1
            return []
        if config.max_date is not None and date_int is not None and date_int > config.max_date:
            stats.date_filtered += 1
            return []

    chunks = chunk_text(text, config)
    if len(chunks) > 1:
        stats.oversized_docs_chunked += 1
        stats.chunks_created += len(chunks)

    records: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        if len(chunk) < config.min_chars:
            stats.too_short += 1
            continue
        if len(chunk) > config.max_chars and not config.chunking.enabled:
            stats.too_long += 1
            continue
        words = word_count(chunk)
        if words < config.min_words:
            stats.too_few_words += 1
            continue
        if looks_low_value(chunk):
            stats.low_value += 1
            continue
        text_hash = sha256_text(chunk)
        records.append(
            {
                "id": f"{row_id}:chunk-{chunk_index:04d}",
                "doc_id": row_id,
                "chunk_index": chunk_index,
                "text": chunk,
                "text_sha256": text_hash,
                "char_count": len(chunk),
                "word_count": words,
                "source_name": source_name,
                "source_type": source_type,
                "source_split": source_split,
                "source_weight": source_weight,
                "source_path_or_dataset": source_path_or_dataset,
                "language": language,
                "license": license_value,
                "date": date_value,
                "collection": collection_value,
                "url": metadata_value(metadata, "url"),
                "source": {"path": source_path_or_dataset, "row": source_row},
                "metadata": metadata,
            }
        )
    return records


def build_normalized_record(*args, **kwargs) -> dict[str, Any] | None:
    records = build_normalized_records(*args, **kwargs)
    return records[0] if records else None


def chunk_text(text: str, config: DataConfig) -> list[str]:
    if not config.chunking.enabled or len(text) <= config.chunking.max_chars:
        return [text]
    target = max(config.min_chars, config.chunking.target_chars)
    max_chars = max(target, config.chunking.max_chars)
    overlap = max(0, min(config.chunking.overlap_chars, target // 2))
    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            preferred = text.rfind("\n\n", start + target // 2, end)
            if preferred == -1:
                preferred = text.rfind(". ", start + target // 2, end)
            if preferred != -1 and preferred > start:
                end = preferred + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)
    return chunks


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def metadata_value(metadata: dict[str, Any], *names: str) -> Any:
    lowered = {key.lower(): value for key, value in metadata.items()}
    for name in names:
        if name in metadata:
            return metadata[name]
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def value_in_allowlist(value: Any, allowlist: list[str]) -> bool:
    if value is None:
        return False
    allowed = {str(item).casefold() for item in allowlist}
    if isinstance(value, list | tuple | set):
        return any(str(item).casefold() in allowed for item in value)
    return str(value).casefold() in allowed


def coerce_year(value: Any) -> int | None:
    if isinstance(value, datetime | date):
        return value.year
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        if len(text) >= 4 and text[:4].isdigit():
            return int(text[:4])
    return None


def iter_normalized_records(paths: list[str | Path]) -> Iterator[dict[str, Any]]:
    import json

    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def render_report(manifest: dict[str, Any]) -> str:
    stats = manifest["stats"]
    lines = [
        "# Perkunas Corpus Normalization Report",
        "",
        f"Input paths: `{', '.join(manifest['input_paths'])}`",
        f"Output shards: `{len(manifest['shards'])}`",
        "",
        "## Stats",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in stats.items())
    lines.extend(["", "## Shards", ""])
    lines.extend(f"- `{shard['path']}` rows={shard['rows']}" for shard in manifest["shards"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize parquet corpus into JSONL shards")
    parser.add_argument("--config", default="training/configs/data.yaml")
    args = parser.parse_args()
    manifest = normalize_corpus(DataConfig.from_yaml(args.config))
    print(f"Normalized rows: {manifest['stats']['written']}")
    print(f"Manifest: {Path(manifest['shards'][0]['path']).parent / 'manifest.json' if manifest['shards'] else 'none'}")


if __name__ == "__main__":
    main()
