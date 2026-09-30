"""Statistically explicit evaluation summaries without SciPy dependencies."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Iterable, Mapping, Sequence

from .providers import CaseRunResult
from .schema import RunStatus


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 1:
        raise ValueError("Wilson interval requires a positive total")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return lower, upper


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _group_summary(
    runs: Sequence[CaseRunResult],
    pricing: Mapping[str, Mapping[str, float]] | None,
) -> dict:
    total = len(runs)
    statuses = Counter(run.final_status.value for run in runs)
    correct = statuses[RunStatus.CORRECT.value]
    wrong = statuses[RunStatus.WRONG.value]
    content_evaluable = correct + wrong
    content_accuracy = correct / content_evaluable if content_evaluable else None
    ci = (
        wilson_interval(correct, content_evaluable)
        if content_evaluable
        else (None, None)
    )
    content_response_total = content_evaluable + statuses[RunStatus.FORMAT_ERROR.value]
    latencies = [
        sum(attempt.latency_ms for attempt in run.attempts) for run in runs
    ]
    prompt_tokens = sum(
        attempt.prompt_tokens or 0 for run in runs for attempt in run.attempts
    )
    completion_tokens = sum(
        attempt.completion_tokens or 0 for run in runs for attempt in run.attempts
    )
    total_tokens = sum(
        attempt.total_tokens or 0 for run in runs for attempt in run.attempts
    )
    model_name = runs[0].model_name if runs else ""
    model_price = pricing.get(model_name) if pricing else None
    if model_price is None:
        estimated_cost = None
        cost_status = "unavailable"
    else:
        estimated_cost = (
            prompt_tokens * float(model_price["input_per_million"])
            + completion_tokens * float(model_price["output_per_million"])
        ) / 1_000_000
        cost_status = "estimated"
    first_truncated = sum(
        run.first_pass_status is RunStatus.TRUNCATED for run in runs
    )
    recovered = sum(run.recovered for run in runs)
    return {
        "total": total,
        "status_counts": dict(statuses),
        "content_evaluable": content_evaluable,
        "content_accuracy": content_accuracy,
        "content_accuracy_ci95": list(ci),
        "end_to_end_success_rate": correct / total if total else None,
        "strict_format_rate": (
            content_evaluable / content_response_total
            if content_response_total
            else None
        ),
        "api_success_rate": (
            1 - statuses[RunStatus.API_ERROR.value] / total if total else None
        ),
        "format_error_rate": (
            statuses[RunStatus.FORMAT_ERROR.value] / total if total else None
        ),
        "api_error_rate": (
            statuses[RunStatus.API_ERROR.value] / total if total else None
        ),
        "truncation_rate": (
            statuses[RunStatus.TRUNCATED.value] / total if total else None
        ),
        "grader_error_rate": (
            statuses[RunStatus.GRADER_ERROR.value] / total if total else None
        ),
        "first_pass_truncation_rate": first_truncated / total if total else None,
        "truncation_recovery_rate": recovered / first_truncated if first_truncated else None,
        "mean_attempts": mean(len(run.attempts) for run in runs) if runs else None,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "mean": mean(latencies) if latencies else None,
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
        "estimated_cost": estimated_cost,
        "cost_status": cost_status,
    }


def summarize_runs(
    runs: Iterable[CaseRunResult],
    *,
    base_case_ids: set[str] | None = None,
    pricing: Mapping[str, Mapping[str, float]] | None = None,
) -> dict:
    selected = [
        run for run in runs if base_case_ids is None or run.case_id in base_case_ids
    ]
    by_model_raw: dict[str, list[CaseRunResult]] = defaultdict(list)
    by_task_raw: dict[str, dict[str, list[CaseRunResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_subject_raw: dict[str, dict[str, list[CaseRunResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in selected:
        by_model_raw[run.model_name].append(run)
        by_task_raw[run.task_type.value][run.model_name].append(run)
        by_subject_raw[run.subject][run.model_name].append(run)
    return {
        "base_case_count": len({run.case_id for run in selected}),
        "result_count": len(selected),
        "by_model": {
            model: _group_summary(items, pricing)
            for model, items in sorted(by_model_raw.items())
        },
        "by_task_type": {
            task: {
                model: _group_summary(items, pricing)
                for model, items in sorted(models.items())
            }
            for task, models in sorted(by_task_raw.items())
        },
        "by_subject": {
            subject: {
                model: _group_summary(items, pricing)
                for model, items in sorted(models.items())
            }
            for subject, models in sorted(by_subject_raw.items())
        },
    }


def mcnemar_exact(
    model_a_runs: Iterable[CaseRunResult],
    model_b_runs: Iterable[CaseRunResult],
) -> dict:
    a = {run.case_id: run for run in model_a_runs if isinstance(run.is_correct, bool)}
    b = {run.case_id: run for run in model_b_runs if isinstance(run.is_correct, bool)}
    paired_ids = sorted(set(a) & set(b))
    a_correct_b_wrong = sum(a[key].is_correct and not b[key].is_correct for key in paired_ids)
    a_wrong_b_correct = sum(not a[key].is_correct and b[key].is_correct for key in paired_ids)
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(a_correct_b_wrong, a_wrong_b_correct) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "paired_cases": len(paired_ids),
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def stability_summary(
    repeated_runs: Iterable[CaseRunResult], expected_runs: int = 3
) -> dict[str, dict]:
    grouped: dict[str, dict[str, list[CaseRunResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in repeated_runs:
        grouped[run.model_name][run.case_id].append(run)
    output = {}
    for model, cases in sorted(grouped.items()):
        per_case = []
        for case_id, runs in sorted(cases.items()):
            labels = [
                run.final_answer
                if run.final_answer is not None
                else run.final_status.value
                for run in runs
            ]
            counts = Counter(labels)
            agreement = max(counts.values()) / len(labels)
            per_case.append(
                {
                    "case_id": case_id,
                    "runs": len(runs),
                    "complete": len(runs) == expected_runs,
                    "answer_agreement": agreement,
                    "outcomes": labels,
                }
            )
        output[model] = {
            "case_count": len(per_case),
            "complete_case_count": sum(item["complete"] for item in per_case),
            "mean_answer_agreement": (
                mean(item["answer_agreement"] for item in per_case)
                if per_case
                else None
            ),
            "cases": per_case,
        }
    return output
