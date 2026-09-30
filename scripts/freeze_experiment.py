"""Create a secret-free manifest for the frozen V2 experiment inputs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from llm_eval.datasets import load_jsonl, sha256_file, validate_source_manifest


DATA_FILES = (
    "benchmark_base_90.jsonl",
    "mmlu_pro_50.jsonl",
    "ifeval_20.jsonl",
    "product_cases_20.jsonl",
    "robustness_30.jsonl",
    "stability_15.jsonl",
    "smoke_3.jsonl",
)


def dependency_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    source_manifest_path = PROJECT_ROOT / "data" / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    validate_source_manifest(source_manifest)
    model_config_path = PROJECT_ROOT / "config" / "models.json"
    model_configs = json.loads(model_config_path.read_text(encoding="utf-8"))
    data_records = []
    for filename in DATA_FILES:
        path = PROJECT_ROOT / "data" / filename
        cases = load_jsonl(path)
        data_records.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "case_count": len(cases),
                "sha256": sha256_file(path),
            }
        )

    key_names = [item["api_key_env"] for item in model_configs]
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "ready_for_smoke"
            if all(os.environ.get(name) for name in key_names)
            else "awaiting_api_keys"
        ),
        "credential_policy": {
            "source": "environment_only",
            "required_environment_variables": key_names,
            "values_recorded": False,
        },
        "models": model_configs,
        "datasets": data_records,
        "source_manifest": {
            "path": "data/source_manifest.json",
            "sha256": sha256_file(source_manifest_path),
            "sources": [
                {
                    "name": source["name"],
                    "resolved_revision": source["resolved_revision"],
                    "license": source["license"],
                }
                for source in source_manifest["sources"]
            ],
        },
        "code": {
            "evaluate_sha256": sha256_file(PROJECT_ROOT / "evaluate.py"),
            "package_files": [
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in sorted((SOURCE_ROOT / "llm_eval").glob("*.py"))
            ],
        },
        "runtime": {
            "python": platform.python_version(),
            "openai": dependency_version("openai"),
            "matplotlib": dependency_version("matplotlib"),
            "datasets": dependency_version("datasets"),
            "nltk": dependency_version("nltk"),
        },
        "run_protocol": {
            "smoke_cases_per_model": 3,
            "base_cases_per_model": 90,
            "stability_cases_per_model": 15,
            "stability_runs": 3,
            "robustness_cases_per_model": 30,
        },
    }
    output = PROJECT_ROOT / "results_v2" / "experiment_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output} ({manifest['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
