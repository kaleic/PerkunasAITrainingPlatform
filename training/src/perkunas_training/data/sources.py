from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pyarrow.parquet as pq

from perkunas_training.config import DataConfig, DatasetSourceConfig
from perkunas_training.data.inspect import choose_text_field, infer_metadata_fields, infer_text_fields


COMMON_TEXT_FIELDS = ("text", "content", "body", "document", "raw", "article")
DOC_ID_FIELDS = ("identifier", "doc_id", "document_id", "id", "url", "source_id", "warc_id")
CANONICAL_METADATA_FIELDS = (
    "language",
    "license",
    "date",
    "timestamp",
    "collection",
    "source",
    "url",
    "identifier",
    "id",
    "title",
    "creator",
    "curator",
    "open_type",
    "language_type",
    "word_count",
    "token_count",
)


@dataclass(slots=True)
class RawSourceRecord:
    source_name: str
    source_type: str
    source_path_or_dataset: str
    source_split: str
    source_weight: float
    doc_id: str
    text: Any
    source_row: int
    metadata: dict[str, Any] = field(default_factory=dict)


def iter_source_records(
    source: DatasetSourceConfig,
    source_split: str,
    config: DataConfig,
) -> Iterator[RawSourceRecord]:
    if source.type == "parquet_local":
        yield from iter_parquet_records(source, source_split, config)
        return
    if source.type == "hf_dataset":
        yield from iter_hf_dataset_records(source, source_split)
        return
    raise ValueError(f"unsupported dataset source type: {source.type}")


def iter_parquet_records(
    source: DatasetSourceConfig,
    source_split: str,
    config: DataConfig,
) -> Iterator[RawSourceRecord]:
    emitted_for_source = 0
    for input_path in source.paths:
        path = Path(input_path)
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        text_field = choose_text_field(
            schema,
            source.text_field or config.text_field,
            infer_text_fields(schema),
        )
        metadata_fields = (
            source.metadata_fields or config.metadata_fields or infer_metadata_fields(schema)
        )
        columns = list(
            dict.fromkeys(
                [text_field, *[field for field in metadata_fields if field in schema.names]]
            )
        )
        row_index = 0
        for batch in pf.iter_batches(batch_size=config.batch_size, columns=columns):
            data = batch.to_pydict()
            for row_offset in range(batch.num_rows):
                if (
                    source.max_records_per_path is not None
                    and row_index >= source.max_records_per_path
                ):
                    break
                if source.max_records is not None and emitted_for_source >= source.max_records:
                    return
                metadata = {
                    field: json_safe(data[field][row_offset])
                    for field in metadata_fields
                    if field in data
                }
                row_id = choose_doc_id(metadata, f"{source.name}:{path.name}:{row_index}")
                yield RawSourceRecord(
                    source_name=source.name,
                    source_type=source.type,
                    source_path_or_dataset=str(path),
                    source_split=source_split,
                    source_weight=source.weight,
                    doc_id=row_id,
                    text=data[text_field][row_offset],
                    source_row=row_index,
                    metadata=metadata,
                )
                emitted_for_source += 1
                row_index += 1
            if (
                source.max_records_per_path is not None
                and row_index >= source.max_records_per_path
            ):
                break


def iter_hf_dataset_records(
    source: DatasetSourceConfig,
    source_split: str,
) -> Iterator[RawSourceRecord]:
    if not source.dataset_name:
        raise ValueError("hf_dataset source requires dataset_name")
    dataset = load_hf_dataset(source)
    if isinstance(dataset, Mapping):
        split_name = source.split or source_split
        if split_name not in dataset:
            available = ", ".join(str(key) for key in dataset.keys())
            raise ValueError(
                f"split {split_name!r} not found in {source.dataset_name!r}; "
                f"available splits: {available}"
            )
        dataset_iterable = dataset[split_name]
    else:
        dataset_iterable = dataset

    for row_index, row in enumerate(dataset_iterable):
        if source.max_records is not None and row_index >= source.max_records:
            break
        if not isinstance(row, Mapping):
            row = {"text": row}
        text_field = choose_mapping_text_field(row, source.text_field)
        metadata = select_metadata(row, text_field, source.metadata_fields)
        row_id = choose_doc_id(metadata, f"{source.name}:{source.split or source_split}:{row_index}")
        yield RawSourceRecord(
            source_name=source.name,
            source_type=source.type,
            source_path_or_dataset=source.dataset_name,
            source_split=source.split or source_split,
            source_weight=source.weight,
            doc_id=row_id,
            text=row.get(text_field),
            source_row=row_index,
            metadata=metadata,
        )


def load_hf_dataset(source: DatasetSourceConfig) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "hf_dataset sources require the Hugging Face datasets package. "
            "Install the training package dependency or run `python -m pip install datasets`."
        ) from exc

    kwargs: dict[str, Any] = {
        "path": source.dataset_name,
        "split": source.split,
        "streaming": source.streaming,
        "trust_remote_code": source.trust_remote_code,
    }
    if source.dataset_config:
        kwargs["name"] = source.dataset_config
    if source.cache_dir:
        kwargs["cache_dir"] = source.cache_dir
    if source.revision:
        kwargs["revision"] = source.revision
    if source.data_files is not None:
        kwargs["data_files"] = source.data_files
    return load_dataset(**kwargs)


def choose_mapping_text_field(row: Mapping[str, Any], configured_text_field: str | None) -> str:
    if configured_text_field:
        if configured_text_field not in row:
            raise ValueError(f"configured text field {configured_text_field!r} not found in HF row")
        return configured_text_field
    for field_name in COMMON_TEXT_FIELDS:
        value = row.get(field_name)
        if isinstance(value, str):
            return field_name
    string_fields = [key for key, value in row.items() if isinstance(value, str)]
    if not string_fields:
        raise ValueError("HF dataset row has no string field candidate for training text")
    scored: list[tuple[int, str]] = []
    for key in string_fields:
        lowered = key.lower()
        score = 0
        if any(hint in lowered for hint in COMMON_TEXT_FIELDS):
            score += 25
        if lowered in CANONICAL_METADATA_FIELDS:
            score -= 20
        scored.append((score, key))
    return sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]


def select_metadata(
    row: Mapping[str, Any], text_field: str, configured_metadata_fields: Iterable[str]
) -> dict[str, Any]:
    if configured_metadata_fields:
        field_names = [field for field in configured_metadata_fields if field in row]
    else:
        field_names = [field for field in row if field != text_field]
    return {field: json_safe(row[field]) for field in field_names}


def choose_doc_id(metadata: Mapping[str, Any], fallback: str) -> str:
    for field_name in DOC_ID_FIELDS:
        value = metadata.get(field_name)
        if value is not None and str(value).strip():
            return str(value)
    return fallback


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value
