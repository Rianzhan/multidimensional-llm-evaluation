from pathlib import Path

import pytest

from llm_eval.datasets import load_jsonl
from llm_eval.ifeval_adapter import grade_ifeval, load_official_evaluator
from llm_eval.schema import RunStatus


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_adapter_matches_official_strict_and_loose_results_for_five_constraints():
    all_cases = load_jsonl(PROJECT_ROOT / "data" / "ifeval_20.jsonl")
    safe_ids = {"3311", "1705", "3718", "2596", "279"}
    cases = [case for case in all_cases if case.source_id in safe_ids]
    assert len(cases) == 5
    official = load_official_evaluator()

    for case in cases:
        response = "Placeholder response without special formatting"
        grader = case.grader
        inp = official.InputExample(
            key=int(case.source_id),
            instruction_id_list=list(grader["instruction_id_list"]),
            prompt=case.prompt,
            kwargs=list(grader["kwargs"]),
        )
        mapping = {case.prompt: response}
        strict = official.test_instruction_following_strict(inp, mapping)
        loose = official.test_instruction_following_loose(inp, mapping)

        result = grade_ifeval(case, response)

        assert result.details["strict"]["prompt_followed"] == strict.follow_all_instructions
        assert result.details["strict"]["instruction_followed"] == strict.follow_instruction_list
        assert result.details["loose"]["prompt_followed"] == loose.follow_all_instructions
        assert result.details["loose"]["instruction_followed"] == loose.follow_instruction_list


def test_strict_prompt_result_controls_primary_status():
    case = next(
        item
        for item in load_jsonl(PROJECT_ROOT / "data" / "ifeval_20.jsonl")
        if item.source_id == "1705"
    )

    result = grade_ifeval(case, "short response")

    assert result.status in {RunStatus.CORRECT, RunStatus.WRONG}
    assert result.is_correct == result.details["strict"]["prompt_followed"]
    assert result.details["evaluator_revision"] == "d36068b845da4c2b24927fee2cea1e6ef98dadda"


def test_missing_nltk_resource_has_actionable_install_command(monkeypatch):
    case = load_jsonl(PROJECT_ROOT / "data" / "ifeval_20.jsonl")[0]

    class BrokenOfficial:
        class InputExample:
            def __init__(self, **_kwargs):
                pass

        @staticmethod
        def test_instruction_following_strict(*_args):
            raise LookupError("punkt resource missing")

    monkeypatch.setattr(
        "llm_eval.ifeval_adapter.load_official_evaluator",
        lambda: BrokenOfficial,
    )

    with pytest.raises(RuntimeError, match=r"python -m nltk\.downloader punkt"):
        grade_ifeval(case, "response")
