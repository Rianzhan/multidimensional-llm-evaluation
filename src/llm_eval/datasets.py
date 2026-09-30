"""Dataset serialization, provenance validation, and pinned downloads."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote

from .schema import EvalCase, validate_case_collection


def write_jsonl(path: Path, cases: Iterable[EvalCase]) -> None:
    validated = validate_case_collection(cases)
    source_keys: set[tuple[str, str]] = set()
    for case in validated:
        key = (case.source_dataset, case.source_id)
        if key in source_keys:
            raise ValueError(f"duplicate source id: {case.source_dataset}/{case.source_id}")
        source_keys.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in validated:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                cases.append(EvalCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL row {line_number} in {path}") from exc
    validated = validate_case_collection(cases)
    source_keys: set[tuple[str, str]] = set()
    for case in validated:
        key = (case.source_dataset, case.source_id)
        if key in source_keys:
            raise ValueError(f"duplicate source id: {case.source_dataset}/{case.source_id}")
        source_keys.add(key)
    return validated


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_sha256(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    opener: Callable[[str], Any] = urllib.request.urlopen,
) -> tuple[Path, str]:
    """Download bytes, validate them, then atomically place the destination."""

    with opener(url) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA256 mismatch for {destination.name}: expected {expected_sha256}, got {digest}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination, digest


def _default_metadata_loader(dataset_id: str) -> Mapping[str, Any]:
    url = f"https://huggingface.co/api/datasets/{quote(dataset_id, safe='/')}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_source_revision(
    dataset_id: str,
    *,
    metadata_loader: Callable[[str], Mapping[str, Any]] = _default_metadata_loader,
) -> str:
    metadata = metadata_loader(dataset_id)
    revision = metadata.get("sha")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"source metadata for {dataset_id} did not contain a revision SHA")
    return revision.strip()


def validate_source_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest sources must be a non-empty list")
    seen: set[str] = set()
    required = (
        "name",
        "source_url",
        "resolved_revision",
        "retrieved_at",
        "license",
        "files",
    )
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("manifest source entry must be an object")
        for field in required:
            if field not in source or source[field] in (None, "", []):
                raise ValueError(f"manifest source is missing {field}")
        name = str(source["name"])
        if name in seen:
            raise ValueError(f"duplicate manifest source: {name}")
        seen.add(name)
        if not isinstance(source["files"], list):
            raise ValueError(f"manifest source {name} files must be a list")
        for item in source["files"]:
            if not isinstance(item, dict) or not item.get("path"):
                raise ValueError(f"manifest source {name} has an invalid file entry")
            digest = item.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in digest)
            ):
                raise ValueError(f"manifest source {name} has an invalid SHA256")
    return manifest
