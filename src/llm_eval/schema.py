"""Canonical case and result schemas shared by every evaluator component."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


class TaskType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    INSTRUCTION_FOLLOWING = "instruction_following"
    RUBRIC = "rubric"


class RunStatus(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    FORMAT_ERROR = "format_error"
    API_ERROR = "api_error"
    TRUNCATED = "truncated"
    GRADER_ERROR = "grader_error"
    PENDING_GRADING = "pending_grading"


GRADER_BY_TASK_TYPE = {
    TaskType.MULTIPLE_CHOICE: "strict_choice",
    TaskType.INSTRUCTION_FOLLOWING: "official_ifeval",
    TaskType.RUBRIC: "product_rubric",
}


def allowed_letters(choice_count: int) -> str:
    """Return the legal answer alphabet for two through ten choices."""

    if not 2 <= choice_count <= 10:
        raise ValueError("choice_count must be between 2 and 10")
    return "ABCDEFGHIJ"[:choice_count]


@dataclass(frozen=True)
class EvalCase:
    id: str
    task_type: TaskType
    subject: str
    difficulty: str
    source_dataset: str
    source_id: str
    source_split: str
    source_url: str
    prompt: str
    choices: tuple[str, ...] = ()
    reference_answer: str | None = None
    grader: Mapping[str, Any] = field(default_factory=dict)
    derived_from: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, TaskType):
            try:
                object.__setattr__(self, "task_type", TaskType(self.task_type))
            except ValueError as exc:
                raise ValueError(f"unknown task_type: {self.task_type}") from exc

        for field_name in (
            "id",
            "subject",
            "difficulty",
            "source_dataset",
            "source_id",
            "source_split",
            "source_url",
            "prompt",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if isinstance(self.choices, str):
            raise ValueError("choices must be a sequence of option strings")
        cleaned_choices = tuple(
            choice.strip()
            for choice in self.choices
            if isinstance(choice, str) and choice.strip().upper() != "N/A"
        )
        if len(cleaned_choices) != sum(
            isinstance(choice, str) for choice in self.choices
        ):
            non_strings = [choice for choice in self.choices if not isinstance(choice, str)]
            if non_strings:
                raise ValueError("choices must contain only strings")
        object.__setattr__(self, "choices", cleaned_choices)
        object.__setattr__(self, "grader", MappingProxyType(dict(self.grader)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

        expected_grader = GRADER_BY_TASK_TYPE[self.task_type]
        if self.grader.get("type") != expected_grader:
            raise ValueError(
                f"grader type for {self.task_type.value} must be {expected_grader}"
            )

        if self.task_type is TaskType.MULTIPLE_CHOICE:
            legal = allowed_letters(len(self.choices))
            if self.reference_answer not in legal:
                raise ValueError(
                    f"reference_answer must be one of {legal} after choice cleaning"
                )
        else:
            if self.choices:
                raise ValueError("choices must be empty for free-text tasks")
            if self.reference_answer is not None:
                raise ValueError("reference_answer must be null for free-text tasks")

    @property
    def allowed_answers(self) -> str:
        if self.task_type is not TaskType.MULTIPLE_CHOICE:
            return ""
        return allowed_letters(len(self.choices))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvalCase":
        fields = dict(value)
        fields["task_type"] = TaskType(fields["task_type"])
        fields["choices"] = tuple(fields.get("choices", ()))
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type.value,
            "subject": self.subject,
            "difficulty": self.difficulty,
            "source_dataset": self.source_dataset,
            "source_id": self.source_id,
            "source_split": self.source_split,
            "source_url": self.source_url,
            "prompt": self.prompt,
            "choices": list(self.choices),
            "reference_answer": self.reference_answer,
            "grader": dict(self.grader),
            "derived_from": self.derived_from,
            "metadata": dict(self.metadata),
        }


def validate_case_collection(cases: Iterable[EvalCase]) -> list[EvalCase]:
    validated = list(cases)
    seen: set[str] = set()
    for case in validated:
        if case.id in seen:
            raise ValueError(f"duplicate case id: {case.id}")
        seen.add(case.id)
    if not validated:
        raise ValueError("case collection must not be empty")
    return validated


@dataclass(frozen=True)
class GradeResult:
    status: RunStatus
    is_correct: bool | None
    normalized_answer: str | None
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
