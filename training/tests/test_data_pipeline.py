from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from perkunas_training.config import DataConfig
from perkunas_training.data.inspect import inspect_parquet_files
from perkunas_training.data.normalize import normalize_corpus
from perkunas_training.utils.io import iter_jsonl


def write_tiny_parquet(path: Path) -> None:
    table = pa.table(
        {
            "identifier": ["a", "b", "c", "d"],
            "language": ["English", "English", "French", "English"],
            "license": ["MIT", "MIT", "CC0", "MIT"],
            "date": [2020, 2021, None, 2023],
            "text": [
                "This is a useful training document with enough words to pass filtering.",
                "short",
                "Voici un document utile avec suffisamment de mots pour le test.",
                "Another useful document. It contains normal whitespace and enough terms.",
            ],
        }
    )
    pq.write_table(table, path)


def test_schema_inspection_identifies_text_and_metadata(tmp_path: Path) -> None:
    parquet = tmp_path / "tiny.parquet"
    write_tiny_parquet(parquet)
    report = inspect_parquet_files([parquet], tmp_path / "reports", batch_size=2)
    assert report["selected_text_field"] == "text"
    assert "language" in report["metadata_fields"]
    assert report["total_rows"] == 4
    assert (tmp_path / "reports" / "parquet_profile.json").exists()


def test_normalization_filters_and_preserves_provenance(tmp_path: Path) -> None:
    parquet = tmp_path / "tiny.parquet"
    write_tiny_parquet(parquet)
    config = DataConfig(
        input_paths=[str(parquet)],
        prepared_dir=str(tmp_path / "prepared"),
        reports_dir=str(tmp_path / "reports"),
        text_field="text",
        metadata_fields=["identifier", "language", "license", "date"],
        min_chars=20,
        min_words=5,
        allowed_languages=["English"],
        output_shard_rows=2,
    )
    manifest = normalize_corpus(config)
    assert manifest["stats"]["seen"] == 4
    assert manifest["stats"]["written"] == 2
    rows = []
    for shard in manifest["shards"]:
        rows.extend(iter_jsonl(shard["path"]))
    assert rows[0]["metadata"]["identifier"] == "a"
    assert rows[0]["source"]["row"] == 0
