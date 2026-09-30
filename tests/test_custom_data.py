import json
from collections import Counter
from pathlib import Path

from llm_eval.datasets import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def test_product_cases_cover_five_categories_with_executable_rubrics():
    cases = load_jsonl(DATA_DIR / "product_cases_20.jsonl")
    counts = Counter(case.subject for case in cases)

    assert len(cases) == 20
    assert counts == {
        "clarification": 4,
        "false_premise": 4,
        "evidence_uncertainty": 4,
        "medical_safety": 4,
        "instruction_conflict": 4,
    }
    for case in cases:
        assert case.grader["rubric_version"] == "1.0"
        assert len(case.grader["criteria"]) == 3
        assert all(item["type"] for item in case.grader["criteria"])
        assert case.grader["critical_failures"]
        assert case.grader["human_review_required"] is True


def test_robustness_has_ten_of_each_transformation_and_six_per_subject():
    cases = load_jsonl(DATA_DIR / "robustness_30.jsonl")

    assert len(cases) == 30
    assert Counter(case.metadata["transformation"] for case in cases) == {
        "option_shuffle": 10,
        "synonym_reframe": 10,
        "irrelevant_information": 10,
    }
    assert set(Counter(case.subject for case in cases).values()) == {6}
    assert all(case.derived_from for case in cases)


def test_robustness_references_existing_originals_and_preserves_answer_mapping():
    originals = {
        case.id: case for case in load_jsonl(DATA_DIR / "mmlu_pro_50.jsonl")
    }
    variants = load_jsonl(DATA_DIR / "robustness_30.jsonl")
    audits = [
        json.loads(line)
        for line in (DATA_DIR / "robustness_audit.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    audit_by_variant = {item["variant_id"]: item for item in audits}

    assert len(audit_by_variant) == 30
    for variant in variants:
        original = originals[variant.derived_from]
        audit = audit_by_variant[variant.id]
        assert audit["original_id"] == original.id
        assert audit["reference_preserved"] is True
        assert audit["audit_status"] == "automated_pass"
        assert audit["human_review_completed"] is False
        correct_text = variant.choices[ord(variant.reference_answer) - ord("A")]
        original_correct_text = original.choices[
            ord(original.reference_answer) - ord("A")
        ]
        assert correct_text == original_correct_text


def test_base_suite_has_exactly_ninety_unique_cases_in_component_order():
    cases = load_jsonl(DATA_DIR / "base_90.jsonl")

    assert len(cases) == 90
    assert len({case.id for case in cases}) == 90
    assert [case.source_dataset for case in cases[:50]] == ["MMLU-Pro"] * 50
    assert [case.source_dataset for case in cases[50:70]] == ["IFEval"] * 20
    assert [case.source_dataset for case in cases[70:]] == ["product_case"] * 20


def test_frozen_base_alias_and_smoke_suite_are_complete():
    base = load_jsonl(DATA_DIR / "base_90.jsonl")
    frozen = load_jsonl(DATA_DIR / "benchmark_base_90.jsonl")
    smoke = load_jsonl(DATA_DIR / "smoke_3.jsonl")

    assert [case.to_dict() for case in frozen] == [case.to_dict() for case in base]
    assert len(smoke) == 3
    assert {case.task_type.value for case in smoke} == {
        "multiple_choice",
        "instruction_following",
        "rubric",
    }
