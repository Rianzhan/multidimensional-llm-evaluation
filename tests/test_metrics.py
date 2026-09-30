import pytest

from llm_eval.metrics import (
    mcnemar_exact,
    stability_summary,
    summarize_runs,
    wilson_interval,
)
from llm_eval.providers import AttemptRecord, CaseRunResult
from llm_eval.schema import RunStatus, TaskType


def result(case_id, model, status, answer=None, latency=100, prompt_tokens=10, completion_tokens=1):
    is_correct = True if status is RunStatus.CORRECT else False if status is RunStatus.WRONG else None
    attempt = AttemptRecord(
        model_name=model,
        model_id=f"{model}-id",
        case_id=case_id,
        attempt_number=1,
        max_tokens=64,
        status=status,
        raw_response=answer,
        normalized_answer=answer,
        is_correct=is_correct,
        finish_reason="length" if status is RunStatus.TRUNCATED else "stop",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency,
        http_status=200,
    )
    return CaseRunResult(
        model_name=model,
        model_id=f"{model}-id",
        case_id=case_id,
        task_type=TaskType.MULTIPLE_CHOICE,
        subject="math",
        attempts=(attempt,),
        first_pass_status=status,
        final_status=status,
        final_answer=answer,
        is_correct=is_correct,
        recovered=False,
    )


def test_content_accuracy_excludes_interface_failures_but_e2e_uses_all_cases():
    runs = [
        result("1", "A", RunStatus.CORRECT, "A"),
        result("2", "A", RunStatus.WRONG, "B"),
        result("3", "A", RunStatus.FORMAT_ERROR),
        result("4", "A", RunStatus.API_ERROR),
        result("5", "A", RunStatus.TRUNCATED),
    ]

    summary = summarize_runs(runs)["by_model"]["A"]

    assert summary["content_evaluable"] == 2
    assert summary["content_accuracy"] == 0.5
    assert summary["end_to_end_success_rate"] == 0.2
    assert summary["format_error_rate"] == 0.2
    assert summary["api_error_rate"] == 0.2
    assert summary["truncation_rate"] == 0.2
    assert sum(summary["status_counts"].values()) == 5


def test_wilson_interval_has_correct_small_sample_boundaries():
    low_zero, high_zero = wilson_interval(0, 10)
    low_all, high_all = wilson_interval(10, 10)
    low_half, high_half = wilson_interval(5, 10)

    assert low_zero == 0
    assert high_zero == pytest.approx(0.2775, abs=0.001)
    assert low_all == pytest.approx(0.7225, abs=0.001)
    assert high_all == 1
    assert (low_half, high_half) == pytest.approx((0.2366, 0.7634), abs=0.001)


def test_mcnemar_uses_only_paired_content_results():
    model_a = [
        result(str(i), "A", RunStatus.CORRECT if i <= 3 else RunStatus.WRONG, "A")
        for i in range(1, 7)
    ]
    model_b = [result(str(i), "B", RunStatus.WRONG, "B") for i in range(1, 7)]
    model_b.append(result("unpaired", "B", RunStatus.CORRECT, "A"))

    comparison = mcnemar_exact(model_a, model_b)

    assert comparison["paired_cases"] == 6
    assert comparison["a_correct_b_wrong"] == 3
    assert comparison["a_wrong_b_correct"] == 0
    assert comparison["p_value"] == pytest.approx(0.25)


def test_stability_reports_three_run_agreement_without_changing_base_count():
    repeats = [
        result("s1", "A", RunStatus.CORRECT, answer) for answer in ("A", "A", "A")
    ] + [
        result("s2", "A", RunStatus.CORRECT, answer) for answer in ("B", "B", "C")
    ]

    stability = stability_summary(repeats, expected_runs=3)["A"]
    base = summarize_runs(
        repeats + [result("base", "A", RunStatus.CORRECT, "A")],
        base_case_ids={"base"},
    )["by_model"]["A"]

    assert stability["case_count"] == 2
    assert stability["complete_case_count"] == 2
    assert stability["mean_answer_agreement"] == pytest.approx(5 / 6)
    assert base["total"] == 1


def test_latency_token_and_unavailable_cost_are_reported():
    runs = [
        result("1", "A", RunStatus.CORRECT, "A", latency=100, prompt_tokens=10),
        result("2", "A", RunStatus.WRONG, "B", latency=300, prompt_tokens=20),
    ]

    summary = summarize_runs(runs)["by_model"]["A"]

    assert summary["latency_ms"]["p50"] == 200
    assert summary["latency_ms"]["p95"] == pytest.approx(290)
    assert summary["tokens"]["prompt"] == 30
    assert summary["tokens"]["completion"] == 2
    assert summary["estimated_cost"] is None
    assert summary["cost_status"] == "unavailable"
