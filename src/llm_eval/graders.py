"""Deterministic graders for benchmark items."""

from __future__ import annotations

from .schema import EvalCase, GradeResult, RunStatus, TaskType


def grade_multiple_choice(case: EvalCase, raw: str | None) -> GradeResult:
    """Grade only an exact single legal uppercase answer letter."""

    if case.task_type is not TaskType.MULTIPLE_CHOICE:
        raise ValueError("grade_multiple_choice requires a multiple-choice case")
    if raw is None:
        return GradeResult(
            status=RunStatus.FORMAT_ERROR,
            is_correct=None,
            normalized_answer=None,
            reason="response content is missing",
        )

    answer = raw.strip()
    if len(answer) != 1 or answer not in case.allowed_answers:
        return GradeResult(
            status=RunStatus.FORMAT_ERROR,
            is_correct=None,
            normalized_answer=None,
            reason=f"expected exactly one of {case.allowed_answers}",
        )
    if answer == case.reference_answer:
        return GradeResult(
            status=RunStatus.CORRECT,
            is_correct=True,
            normalized_answer=answer,
        )
    return GradeResult(
        status=RunStatus.WRONG,
        is_correct=False,
        normalized_answer=answer,
    )
