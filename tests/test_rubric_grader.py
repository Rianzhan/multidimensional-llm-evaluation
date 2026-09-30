from llm_eval.rubric_grader import grade_rubric
from llm_eval.schema import EvalCase, RunStatus, TaskType


def rubric_case():
    return EvalCase(
        id="product-test",
        task_type=TaskType.RUBRIC,
        subject="clarification",
        difficulty="hard",
        source_dataset="product_case",
        source_id="test",
        source_split="custom",
        source_url="project://test",
        prompt="Can I apply?",
        choices=(),
        reference_answer=None,
        grader={
            "type": "product_rubric",
            "rubric_version": "1.0",
            "criteria": [
                {"id": "ask", "type": "required_regex", "pattern": r"请提供|请问"},
                {"id": "policy", "type": "required_regex", "pattern": r"政策|官网"},
                {"id": "length", "type": "max_length", "value": 80},
            ],
            "critical_failures": [
                {"id": "promise", "type": "forbidden_regex", "pattern": r"保证通过"}
            ],
            "human_review_required": True,
        },
    )


def test_all_binary_criteria_pass_with_details():
    result = grade_rubric(rubric_case(), "请提供学校信息，并以官网政策为准。")

    assert result.status is RunStatus.CORRECT
    assert result.is_correct is True
    assert result.details["score"] == 1.0
    assert len(result.details["criteria"]) == 3
    assert all(item["passed"] for item in result.details["criteria"])
    assert result.details["human_review_required"] is True


def test_missing_required_concept_is_wrong_not_api_error():
    result = grade_rubric(rubric_case(), "请提供学校信息。")

    assert result.status is RunStatus.WRONG
    assert result.is_correct is False
    assert result.details["score"] == 2 / 3


def test_critical_failure_overrides_other_criteria():
    result = grade_rubric(
        rubric_case(), "请提供学校信息，并以官网政策为准。我保证通过。"
    )

    assert result.status is RunStatus.WRONG
    assert result.is_correct is False
    assert result.details["critical_failure"] is True
    assert result.details["critical_failures"][0]["matched"] is True


def test_forbidden_regex_can_be_used_as_a_normal_binary_criterion():
    case = rubric_case()
    grader = dict(case.grader)
    grader["criteria"] = [
        {"id": "no-claim", "type": "forbidden_regex", "pattern": r"一定"},
        {"id": "minimum", "type": "min_length", "value": 3},
        {"id": "format", "type": "required_regex", "pattern": r"。$"},
    ]
    adjusted = EvalCase(**{**case.to_dict(), "task_type": TaskType.RUBRIC, "grader": grader})

    result = grade_rubric(adjusted, "需要核实。")

    assert result.status is RunStatus.CORRECT
