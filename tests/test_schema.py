import pytest

from llm_eval.schema import (
    EvalCase,
    TaskType,
    allowed_letters,
    validate_case_collection,
)


def choice_case(**overrides):
    values = {
        "id": "mmlu-pro-math-001",
        "task_type": TaskType.MULTIPLE_CHOICE,
        "subject": "math",
        "difficulty": "hard",
        "source_dataset": "MMLU-Pro",
        "source_id": "001",
        "source_split": "test",
        "source_url": "https://example.test/source",
        "prompt": "Pick the correct option.",
        "choices": ("one", "two", "three", "four"),
        "reference_answer": "D",
        "grader": {"type": "strict_choice"},
    }
    values.update(overrides)
    return EvalCase(**values)


def test_allowed_letters_supports_four_and_ten_choices():
    assert allowed_letters(4) == "ABCD"
    assert allowed_letters(10) == "ABCDEFGHIJ"


def test_na_choices_are_removed_before_allowed_range_is_computed():
    case = choice_case(
        choices=("one", "two", "three", "four", "N/A", "  N/A  "),
        reference_answer="D",
    )

    assert case.choices == ("one", "two", "three", "four")
    assert case.allowed_answers == "ABCD"


def test_reference_answer_must_be_inside_cleaned_choice_range():
    with pytest.raises(ValueError, match="reference_answer"):
        choice_case(choices=("one", "two", "N/A"), reference_answer="C")


def test_grader_must_match_task_type():
    with pytest.raises(ValueError, match="grader"):
        choice_case(grader={"type": "official_ifeval"})


def test_duplicate_case_ids_are_rejected():
    first = choice_case()
    second = choice_case(source_id="002")

    with pytest.raises(ValueError, match="duplicate case id"):
        validate_case_collection([first, second])


def test_free_text_case_rejects_choices_and_reference_answer():
    with pytest.raises(ValueError, match="choices"):
        EvalCase(
            id="ifeval-001",
            task_type=TaskType.INSTRUCTION_FOLLOWING,
            subject="instruction_following",
            difficulty="hard",
            source_dataset="IFEval",
            source_id="001",
            source_split="test",
            source_url="https://example.test/ifeval",
            prompt="Write exactly three sentences.",
            choices=("unexpected",),
            reference_answer=None,
            grader={"type": "official_ifeval"},
        )
