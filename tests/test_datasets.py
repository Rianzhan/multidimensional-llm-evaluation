import hashlib
import io
import json
from pathlib import Path

import pytest

from llm_eval.datasets import (
    download_with_sha256,
    load_jsonl,
    resolve_source_revision,
    validate_source_manifest,
    write_jsonl,
)
from llm_eval.schema import EvalCase, TaskType


def make_case(case_id="case-1", source_id="source-1"):
    return EvalCase(
        id=case_id,
        task_type=TaskType.MULTIPLE_CHOICE,
        subject="math",
        difficulty="hard",
        source_dataset="fixture",
        source_id=source_id,
        source_split="test",
        source_url="https://example.test/source",
        prompt="Question?",
        choices=("a", "b", "c", "d"),
        reference_answer="A",
        grader={"type": "strict_choice"},
    )


def test_jsonl_round_trip_preserves_valid_cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_jsonl(path, [make_case()])

    loaded = load_jsonl(path)

    assert loaded == [make_case()]


def test_jsonl_reader_does_not_split_unicode_line_separator_inside_prompt(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_jsonl(path, [make_case()])
    text = path.read_text(encoding="utf-8").replace("Question?", "left\u2028right")
    path.write_text(text, encoding="utf-8")

    assert load_jsonl(path)[0].prompt == "left\u2028right"


def test_jsonl_rejects_duplicate_case_or_source_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [
        make_case("case-1", "same").to_dict(),
        make_case("case-2", "same").to_dict(),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source id"):
        load_jsonl(path)


def test_manifest_requires_traceability_and_checksum_fields():
    manifest = {
        "sources": [
            {
                "name": "fixture",
                "source_url": "https://example.test/data",
                "resolved_revision": "abc123",
                "retrieved_at": "2026-09-28T00:00:00Z",
                "license": "Apache-2.0",
                "files": [{"path": "fixture.jsonl", "sha256": "a" * 64}],
            }
        ]
    }

    assert validate_source_manifest(manifest) == manifest

    broken = json.loads(json.dumps(manifest))
    del broken["sources"][0]["license"]
    with pytest.raises(ValueError, match="license"):
        validate_source_manifest(broken)


def test_download_validates_sha256_without_network(tmp_path):
    payload = b"official fixture\n"
    expected = hashlib.sha256(payload).hexdigest()

    path, digest = download_with_sha256(
        "https://example.test/data",
        tmp_path / "source.bin",
        expected_sha256=expected,
        opener=lambda _url: io.BytesIO(payload),
    )

    assert path.read_bytes() == payload
    assert digest == expected


def test_download_rejects_corrupt_checksum(tmp_path):
    with pytest.raises(ValueError, match="SHA256"):
        download_with_sha256(
            "https://example.test/data",
            tmp_path / "source.bin",
            expected_sha256="0" * 64,
            opener=lambda _url: io.BytesIO(b"different"),
        )


def test_resolve_source_revision_uses_injected_metadata_loader():
    revision = resolve_source_revision(
        "TIGER-Lab/MMLU-Pro",
        metadata_loader=lambda dataset_id: {"id": dataset_id, "sha": "deadbeef"},
    )

    assert revision == "deadbeef"
