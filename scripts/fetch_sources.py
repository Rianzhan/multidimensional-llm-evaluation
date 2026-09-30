"""Fetch and freeze MMLU-Pro plus the official Google IFEval implementation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from llm_eval.datasets import (
    download_with_sha256,
    resolve_source_revision,
    sha256_file,
    validate_source_manifest,
)


MMLU_DATASET_ID = "TIGER-Lab/MMLU-Pro"
IFEVAL_REVISION = "d36068b845da4c2b24927fee2cea1e6ef98dadda"
IFEVAL_BASE_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    f"{IFEVAL_REVISION}/instruction_following_eval"
)
IFEVAL_FILES = (
    "data/input_data.jsonl",
    "evaluation_lib.py",
    "evaluation_main.py",
    "instructions.py",
    "instructions_registry.py",
    "instructions_util.py",
    "requirements.txt",
)


def fetch_mmlu_pro(source_dir: Path) -> tuple[dict, list[dict]]:
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "hf_cache"))
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("请先运行 pip install -r requirements.txt 安装 datasets") from exc

    metadata_path = source_dir / "mmlu_pro_api.json"
    test_path = source_dir / "mmlu_pro_test.parquet"
    validation_path = source_dir / "mmlu_pro_validation.parquet"
    if metadata_path.exists() and test_path.exists() and validation_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        revision = str(metadata["sha"])
        dataset = load_dataset(
            "parquet",
            data_files={
                "test": str(test_path),
                "validation": str(validation_path),
            },
        )
    else:
        revision = resolve_source_revision(MMLU_DATASET_ID)
        dataset = load_dataset(MMLU_DATASET_ID, revision=revision)
    rows: list[dict] = []
    fingerprints: dict[str, str | None] = {}
    for split_name in sorted(dataset):
        split = dataset[split_name]
        fingerprints[split_name] = getattr(split, "_fingerprint", None)
        for row_index, row in enumerate(split):
            value = dict(row)
            value["_split"] = split_name
            value["_row_index"] = row_index
            rows.append(value)

    destination = source_dir / "mmlu_pro.jsonl"
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    local_files = [destination]
    local_files.extend(
        path
        for path in (metadata_path, test_path, validation_path)
        if path.exists()
    )
    file_records = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in local_files
    ]
    return (
        {
            "name": "MMLU-Pro",
            "source_url": f"https://huggingface.co/datasets/{MMLU_DATASET_ID}",
            "resolved_revision": revision,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "license": "Apache-2.0",
            "fingerprints": fingerprints,
            "files": file_records,
        },
        rows,
    )


def fetch_ifeval(source_dir: Path) -> dict:
    destination_root = PROJECT_ROOT / "third_party" / "ifeval"
    records = []
    for relative in IFEVAL_FILES:
        destination = destination_root / relative
        if destination.exists():
            digest = sha256_file(destination)
        else:
            destination, digest = download_with_sha256(
                f"{IFEVAL_BASE_URL}/{relative}", destination
            )
        records.append(
            {
                "path": destination.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": digest,
            }
        )
    return {
        "name": "IFEval",
        "source_url": (
            "https://github.com/google-research/google-research/tree/"
            f"{IFEVAL_REVISION}/instruction_following_eval"
        ),
        "resolved_revision": IFEVAL_REVISION,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license": "Apache-2.0",
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-mmlu-pro",
        action="store_true",
        help="Only refresh the pinned IFEval source files.",
    )
    args = parser.parse_args()
    source_dir = PROJECT_ROOT / "data" / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    if not args.skip_mmlu_pro:
        mmlu_manifest, _ = fetch_mmlu_pro(source_dir)
        sources.append(mmlu_manifest)
    sources.append(fetch_ifeval(source_dir))
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
    }
    validate_source_manifest(manifest)
    destination = PROJECT_ROOT / "data" / "source_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
