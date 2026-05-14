from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any

from perkunas_training.config import DataConfig
from perkunas_training.utils.hashing import hamming64, sha256_text, simhash64
from perkunas_training.utils.io import ensure_dir, iter_jsonl, write_json, write_jsonl


@dataclass(slots=True)
class SimHashIndex:
    band_bits: int = 16
    hamming_threshold: int = 3
    buckets: dict[tuple[int, int], list[tuple[int, str]]] = field(default_factory=lambda: defaultdict(list))
    signatures: dict[str, int] = field(default_factory=dict)

    def find_near_duplicate(self, doc_id: str, signature: int) -> str | None:
        candidates: set[str] = set()
        bands = 64 // self.band_bits
        mask = (1 << self.band_bits) - 1
        for band in range(bands):
            key = (band, (signature >> (band * self.band_bits)) & mask)
            candidates.update(candidate_id for _, candidate_id in self.buckets.get(key, []))
        for candidate_id in candidates:
            candidate_sig = self.signatures[candidate_id]
            if hamming64(signature, candidate_sig) <= self.hamming_threshold:
                return candidate_id
        return None

    def add(self, doc_id: str, signature: int) -> None:
        self.signatures[doc_id] = signature
        bands = 64 // self.band_bits
        mask = (1 << self.band_bits) - 1
        for band in range(bands):
            key = (band, (signature >> (band * self.band_bits)) & mask)
            self.buckets[key].append((signature, doc_id))


def deduplicate_corpus(
    input_glob: str,
    output_dir: str | Path,
    *,
    shard_rows: int = 5000,
    approximate: bool = True,
    hamming_threshold: int = 3,
    max_simhash_tokens: int = 2048,
    max_simhash_chars: int = 120_000,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    manifest_path = output_dir / "manifest.json"
    if resume and manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete"):
            return manifest

    input_paths = sorted(glob(input_glob))
    if not input_paths:
        raise FileNotFoundError(f"no normalized shards matched {input_glob}")

    seen_hashes: dict[str, str] = {}
    simhash_index = SimHashIndex(hamming_threshold=hamming_threshold)
    duplicate_examples: list[dict[str, Any]] = []
    stats = Counter()
    shard_manifest: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index, current
        if not current:
            return
        path = output_dir / f"dedup_{shard_index:05d}.jsonl"
        count = write_jsonl(path, current)
        shard_manifest.append({"path": str(path), "rows": count})
        shard_index += 1
        current = []

    for input_path in input_paths:
        for record in iter_jsonl(input_path):
            stats["seen"] += 1
            text = record["text"]
            doc_hash = record.get("text_sha256") or sha256_text(text)
            if doc_hash in seen_hashes:
                stats["exact_duplicates"] += 1
                maybe_record_duplicate(duplicate_examples, record, seen_hashes[doc_hash], "exact")
                continue
            signature = simhash64(
                sample_for_simhash(text, max_chars=max_simhash_chars),
                max_tokens=max_simhash_tokens,
            )
            if approximate:
                near_id = simhash_index.find_near_duplicate(record["id"], signature)
                if near_id is not None:
                    stats["approx_duplicates"] += 1
                    maybe_record_duplicate(duplicate_examples, record, near_id, "simhash")
                    continue
            seen_hashes[doc_hash] = record["id"]
            simhash_index.add(record["id"], signature)
            record["simhash64"] = str(signature)
            current.append(record)
            stats["written"] += 1
            if len(current) >= shard_rows:
                flush()
    flush()

    manifest = {
        "stage": "dedup",
        "complete": True,
        "input_glob": input_glob,
        "input_paths": input_paths,
        "approximate": approximate,
        "hamming_threshold": hamming_threshold,
        "max_simhash_tokens": max_simhash_tokens,
        "max_simhash_chars": max_simhash_chars,
        "stats": dict(stats),
        "duplicate_examples": duplicate_examples,
        "shards": shard_manifest,
    }
    write_json(manifest_path, manifest)
    (output_dir / "dedup_report.md").write_text(render_report(manifest), encoding="utf-8")
    return manifest


def maybe_record_duplicate(
    examples: list[dict[str, Any]], record: dict[str, Any], matched_id: str, mode: str
) -> None:
    if len(examples) >= 100:
        return
    examples.append(
        {
            "id": record.get("id"),
            "matched_id": matched_id,
            "mode": mode,
            "char_count": record.get("char_count"),
            "source": record.get("source"),
        }
    )


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Perkunas Deduplication Report",
        "",
        f"Approximate deduplication: `{manifest['approximate']}`",
        f"SimHash Hamming threshold: `{manifest['hamming_threshold']}`",
        f"Max SimHash tokens per document: `{manifest['max_simhash_tokens']}`",
        f"Max SimHash chars per document: `{manifest['max_simhash_chars']}`",
        "",
        "## Stats",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in manifest["stats"].items())
    lines.extend(["", "## Output Shards", ""])
    lines.extend(f"- `{shard['path']}` rows={shard['rows']}" for shard in manifest["shards"])
    return "\n".join(lines) + "\n"


def sample_for_simhash(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n" + text[-tail:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate normalized Perkunas corpus shards")
    parser.add_argument("--config", default="training/configs/data.yaml")
    parser.add_argument("--input-glob")
    parser.add_argument("--output-dir")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--hamming-threshold", type=int)
    parser.add_argument("--max-simhash-tokens", type=int)
    parser.add_argument("--max-simhash-chars", type=int)
    args = parser.parse_args()
    config = DataConfig.from_yaml(args.config)
    input_glob = args.input_glob or str(Path(config.prepared_dir) / "normalized_*.jsonl")
    output_dir = args.output_dir or config.dedup_dir
    approximate = config.dedup.approximate and not args.exact_only
    manifest = deduplicate_corpus(
        input_glob,
        output_dir,
        shard_rows=config.output_shard_rows,
        approximate=approximate,
        hamming_threshold=args.hamming_threshold or config.dedup.hamming_threshold,
        max_simhash_tokens=args.max_simhash_tokens or config.dedup.max_simhash_tokens,
        max_simhash_chars=args.max_simhash_chars or config.dedup.max_simhash_chars,
        resume=config.resume,
    )
    print(f"Deduplicated rows: {manifest['stats'].get('written', 0)}")
    print(f"Manifest: {Path(output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
