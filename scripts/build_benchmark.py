"""Build deterministic MMLU-Pro and IFEval evaluation components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from llm_eval.benchmark import select_ifeval, select_mmlu_pro, select_stability
from llm_eval.custom_cases import build_product_cases, build_robustness_cases
from llm_eval.datasets import load_jsonl, write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_data_card() -> None:
    content = """# Evaluation Data Card

## Purpose

This frozen suite compares GLM, Kimi, and DeepSeek on difficult knowledge,
instruction following, and product-behavior tasks. Selection is fixed before
model execution and never uses historical model errors.

## Public benchmark components

- **MMLU-Pro (50):** 10 each from math, computer science, physics, health, and
  law. Candidates are ranked by a model-independent multi-step/length proxy;
  one variable-choice item is retained per subject when available.
- **IFEval (20):** URL-free prompts selected greedily for instruction-family
  coverage with seed `20260928`.
- **Stability diagnostic (15):** three frozen MMLU-Pro items per subject. These
  are repeated three times and are not added again to base accuracy.

## Provenance and licenses

Exact revisions, retrieval times, and SHA256 values are recorded in
`source_manifest.json`. MMLU-Pro and Google Research IFEval are attributed
under Apache-2.0 in `third_party/NOTICE.md`.

## Limitations

The 50-question knowledge slice is intentionally small and optimized for
difficulty, so confidence intervals must accompany rankings. Public benchmarks
may have appeared in model training data. IFEval measures verifiable surface
constraints rather than factual quality. Product cases are reported separately.
"""
    (PROJECT_ROOT / "data" / "DATA_CARD.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("mmlu-pro", "ifeval"),
        default=("mmlu-pro", "ifeval"),
    )
    parser.add_argument("--all", action="store_true", help="Build all V2 datasets.")
    args = parser.parse_args()
    data_dir = PROJECT_ROOT / "data"

    mmlu_cases = None
    if "mmlu-pro" in args.components:
        rows = read_jsonl(data_dir / "sources" / "mmlu_pro.jsonl")
        mmlu_cases = select_mmlu_pro(rows)
        write_jsonl(data_dir / "mmlu_pro_50.jsonl", mmlu_cases)
        write_jsonl(data_dir / "stability_15.jsonl", select_stability(mmlu_cases))
        print(f"MMLU-Pro: {len(mmlu_cases)}; stability: 15")

    if "ifeval" in args.components:
        rows = read_jsonl(PROJECT_ROOT / "third_party" / "ifeval" / "data" / "input_data.jsonl")
        ifeval_cases = select_ifeval(rows)
        write_jsonl(data_dir / "ifeval_20.jsonl", ifeval_cases)
        print(f"IFEval: {len(ifeval_cases)}")

    if args.all:
        if mmlu_cases is None:
            mmlu_cases = load_jsonl(data_dir / "mmlu_pro_50.jsonl")
        product_cases = build_product_cases()
        robustness_cases, audits = build_robustness_cases(mmlu_cases)
        write_jsonl(data_dir / "product_cases_20.jsonl", product_cases)
        write_jsonl(data_dir / "robustness_30.jsonl", robustness_cases)
        with (data_dir / "robustness_audit.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for item in audits:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        base_cases = (
            load_jsonl(data_dir / "mmlu_pro_50.jsonl")
            + load_jsonl(data_dir / "ifeval_20.jsonl")
            + product_cases
        )
        write_jsonl(data_dir / "base_90.jsonl", base_cases)
        write_jsonl(data_dir / "benchmark_base_90.jsonl", base_cases)
        smoke_cases = [
            load_jsonl(data_dir / "mmlu_pro_50.jsonl")[0],
            load_jsonl(data_dir / "ifeval_20.jsonl")[0],
            product_cases[0],
        ]
        write_jsonl(data_dir / "smoke_3.jsonl", smoke_cases)
        print("Product cases: 20; robustness: 30; base suite: 90")

    write_data_card()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
