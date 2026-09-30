from llm_eval.graders import grade_multiple_choice
from llm_eval.prompts import build_prompt
from llm_eval.schema import EvalCase, RunStatus, TaskType


def make_case(choice_count: int, reference_answer: str) -> EvalCase:
    return EvalCase(
        id=f"choice-{choice_count}",
        task_type=TaskType.MULTIPLE_CHOICE,
        subject="test",
        difficulty="hard",
        source_dataset="fixture",
        source_id=str(choice_count),
        source_split="test",
        source_url="https://example.test",
        prompt="Which option is correct?",
        choices=tuple(f"option {index}" for index in range(choice_count)),
        reference_answer=reference_answer,
        grader={"type": "strict_choice"},
    )


def test_j_is_valid_for_a_ten_choice_question():
    result = grade_multiple_choice(make_case(10, "J"), "J")

    assert result.status is RunStatus.CORRECT
    assert result.is_correct is True
    assert result.normalized_answer == "J"


def test_k_is_out_of_range_for_a_ten_choice_question():
    result = grade_multiple_choice(make_case(10, "J"), "K")

    assert result.status is RunStatus.FORMAT_ERROR
    assert result.is_correct is None


def test_explanation_is_rejected_even_when_it_starts_with_correct_letter():
    result = grade_multiple_choice(make_case(4, "A"), "A. explanation")

    assert result.status is RunStatus.FORMAT_ERROR
    assert result.is_correct is None


def test_legal_wrong_answer_is_distinct_from_format_error():
    result = grade_multiple_choice(make_case(4, "A"), "B")

    assert result.status is RunStatus.WRONG
    assert result.is_correct is False
    assert result.normalized_answer == "B"


def test_only_surrounding_whitespace_is_removed():
    assert grade_multiple_choice(make_case(4, "C"), "\n C \t").status is RunStatus.CORRECT
    assert grade_multiple_choice(make_case(4, "C"), "answer: C").status is RunStatus.FORMAT_ERROR


def test_choice_prompt_lists_dynamic_range_and_all_labels():
    messages = build_prompt(make_case(10, "J"))

    assert messages[0]["role"] == "user"
    assert "A-J" in messages[0]["content"]
    assert "A. option 0" in messages[0]["content"]
    assert "J. option 9" in messages[0]["content"]


def test_free_text_prompt_is_not_wrapped_as_a_choice_question():
    case = EvalCase(
        id="ifeval-001",
        task_type=TaskType.INSTRUCTION_FOLLOWING,
        subject="instruction_following",
        difficulty="hard",
        source_dataset="IFEval",
        source_id="001",
        source_split="test",
        source_url="https://example.test",
        prompt="Write exactly three sentences.",
        choices=(),
        reference_answer=None,
        grader={"type": "official_ifeval"},
    )

    assert build_prompt(case) == [
        {"role": "user", "content": "Write exactly three sentences."}
    ]
