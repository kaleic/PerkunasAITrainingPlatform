from __future__ import annotations

from pathlib import Path

from perkunas_training.config import ChunkingConfig, DataConfig, DatasetSourceConfig, TokenizerConfig
from perkunas_training.data import sources
from perkunas_training.data.normalize import normalize_corpus
from perkunas_training.tokenizer.train_tokenizer import train_perkunas_tokenizer
from perkunas_training.utils.io import iter_jsonl


def fake_common_corpus_rows(count: int = 4) -> list[dict[str, object]]:
    base_text = (
        "Perkunas needs Common Corpus style records with enough useful natural language "
        "to exercise filtering, chunking, provenance, and tokenizer training. "
    )
    return [
        {
            "identifier": f"common-{idx}",
            "text": base_text * (4 if idx == 0 else 2),
            "language": "English" if idx != 2 else "French",
            "license": "CC0",
            "collection": "test_collection",
            "date": "2021-01-01",
            "url": f"https://example.test/doc/{idx}",
        }
        for idx in range(count)
    ]


def test_hf_dataset_adapter_maps_common_record_shape(monkeypatch) -> None:
    monkeypatch.setattr(sources, "load_hf_dataset", lambda source: fake_common_corpus_rows(2))
    source = DatasetSourceConfig(
        name="common_corpus_hf",
        type="hf_dataset",
        dataset_name="PleIAs/common_corpus",
        split="train",
        streaming=True,
        text_field="text",
    )

    records = list(iter(sources.iter_source_records(source, "train", DataConfig(input_paths=[]))))

    assert len(records) == 2
    assert records[0].source_type == "hf_dataset"
    assert records[0].source_path_or_dataset == "PleIAs/common_corpus"
    assert records[0].doc_id == "common-0"
    assert records[0].metadata["collection"] == "test_collection"


def test_hf_normalization_filters_and_chunks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sources, "load_hf_dataset", lambda source: fake_common_corpus_rows(3))
    config = DataConfig(
        input_paths=[],
        datasets=[
            DatasetSourceConfig(
                name="common_corpus_hf",
                type="hf_dataset",
                dataset_name="PleIAs/common_corpus",
                split="train",
                streaming=True,
                text_field="text",
            )
        ],
        prepared_dir=str(tmp_path / "prepared"),
        reports_dir=str(tmp_path / "reports"),
        min_chars=40,
        min_words=6,
        max_chars=1000,
        language_allowlist=["English"],
        collection_blocklist=["blocked_collection"],
        chunking=ChunkingConfig(enabled=True, target_chars=120, max_chars=180, overlap_chars=20),
        output_shard_rows=10,
        resume=False,
    )

    manifest = normalize_corpus(config)
    rows = [row for shard in manifest["shards"] for row in iter_jsonl(shard["path"])]

    assert manifest["stats"]["seen"] == 3
    assert manifest["stats"]["language_filtered"] == 1
    assert manifest["stats"]["oversized_docs_chunked"] >= 1
    assert rows
    assert rows[0]["source_type"] == "hf_dataset"
    assert rows[0]["source_path_or_dataset"] == "PleIAs/common_corpus"
    assert rows[0]["collection"] == "test_collection"
    assert rows[0]["metadata"]["identifier"] == "common-0"


def test_hf_normalized_output_can_train_perkunas_tokenizer(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sources, "load_hf_dataset", lambda source: fake_common_corpus_rows(30))
    config = DataConfig(
        input_paths=[],
        datasets=[
            DatasetSourceConfig(
                name="common_corpus_hf",
                type="hf_dataset",
                dataset_name="PleIAs/common_corpus",
                split="train",
                streaming=True,
                text_field="text",
            )
        ],
        prepared_dir=str(tmp_path / "prepared"),
        reports_dir=str(tmp_path / "reports"),
        min_chars=40,
        min_words=6,
        language_allowlist=["English"],
        chunking=ChunkingConfig(enabled=False),
        output_shard_rows=50,
        resume=False,
    )
    manifest = normalize_corpus(config)
    assert manifest["stats"]["written"] > 0

    result = train_perkunas_tokenizer(
        TokenizerConfig(
            input_glob=str(tmp_path / "prepared" / "*.jsonl"),
            output_dir=str(tmp_path / "tokenizer"),
            vocab_size=300,
            min_frequency=1,
            sample_size=10,
        )
    )

    assert Path(result["tokenizer_json"]).exists()
    assert result["evaluation"]["sample_count"] == 10
