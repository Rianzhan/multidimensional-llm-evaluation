"""Deterministic binary-criterion grading for product behavior cases."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .schema import EvalCase, GradeResult, RunStatus, TaskType


def _evaluate_criterion(criterion: Mapping[str, Any], response: str) -> bool:
    criterion_type = criterion.get("type")
    if criterion_type == "required_regex":
        return re.search(str(criterion["pattern"]), response, re.IGNORECASE) is not None
    if criterion_type == "forbidden_regex":
        return re.search(str(criterion["pattern"]), response, re.IGNORECASE) is None
    if criterion_type == "max_length":
        return len(response) <= int(criterion["value"])
    if criterion_type == "min_length":
        return len(response) >= int(criterion["value"])
    if criterion_type == "exact_format":
        return re.fullmatch(str(criterion["pattern"]), response) is not None
    raise ValueError(f"unsupported rubric criterion type: {criterion_type}")


def grade_rubric(case: EvalCase, response: str) -> GradeResult:
    if case.task_type is not TaskType.RUBRIC:
        raise ValueError("grade_rubric requires a rubric case")
    grader = case.grader
    criteria_details = []
    for criterion in grader.get("criteria", []):
        passed = _evaluate_criterion(criterion, response)
        criteria_details.append(
            {
                "id": criterion.get("id"),
                "type": criterion.get("type"),
                "description": criterion.get("description"),
                "passed": passed,
            }
        )

    critical_details = []
    for failure in grader.get("critical_failures", []):
        if failure.get("type") != "forbidden_regex":
            raise ValueError("critical failures must use forbidden_regex")
        matched = re.search(
            str(failure["pattern"]), response, re.IGNORECASE
        ) is not None
        critical_details.append(
            {
                "id": failure.get("id"),
                "description": failure.get("description"),
                "matched": matched,
            }
        )

    passed_count = sum(item["passed"] for item in criteria_details)
    total = len(criteria_details)
    score = passed_count / total if total else 0.0
    critical_failure = any(item["matched"] for item in critical_details)
    passed = bool(total) and passed_count == total and not critical_failure
    return GradeResult(
        status=RunStatus.CORRECT if passed else RunStatus.WRONG,
        is_correct=passed,
        normalized_answer=None,
        details={
            "rubric_version": grader.get("rubric_version"),
            "score": score,
            "criteria": criteria_details,
            "critical_failure": critical_failure,
            "critical_failures": critical_details,
            "human_review_required": bool(grader.get("human_review_required", False)),
        },
    )
