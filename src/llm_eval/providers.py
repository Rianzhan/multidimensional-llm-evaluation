"""OpenAI-compatible provider calls with auditable attempt evidence."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from .config import ModelConfig
from .graders import grade_multiple_choice
from .ifeval_adapter import grade_ifeval
from .prompts import build_prompt
from .rubric_grader import grade_rubric
from .schema import EvalCase, RunStatus, TaskType


ClientFactory = Callable[..., Any]
Clock = Callable[[], float]


@dataclass(frozen=True)
class AttemptRecord:
    model_name: str
    model_id: str
    case_id: str
    attempt_number: int
    max_tokens: int
    status: RunStatus
    raw_response: str | None
    normalized_answer: str | None
    is_correct: bool | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    http_status: int | None
    error_type: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    wait_seconds: float = 0.0
    grader_details: dict[str, Any] | None = None
    retry_headers: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AttemptRecord":
        fields = dict(value)
        fields["status"] = RunStatus(fields["status"])
        return cls(**fields)


@dataclass(frozen=True)
class CaseRunResult:
    model_name: str
    model_id: str
    case_id: str
    task_type: TaskType
    subject: str
    attempts: tuple[AttemptRecord, ...]
    first_pass_status: RunStatus
    final_status: RunStatus
    final_answer: str | None
    is_correct: bool | None
    recovered: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "case_id": self.case_id,
            "task_type": self.task_type.value,
            "subject": self.subject,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "first_pass_status": self.first_pass_status.value,
            "final_status": self.final_status.value,
            "final_answer": self.final_answer,
            "is_correct": self.is_correct,
            "recovered": self.recovered,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CaseRunResult":
        return cls(
            model_name=value["model_name"],
            model_id=value["model_id"],
            case_id=value["case_id"],
            task_type=TaskType(value["task_type"]),
            subject=value["subject"],
            attempts=tuple(
                AttemptRecord.from_dict(item) for item in value.get("attempts", [])
            ),
            first_pass_status=RunStatus(value["first_pass_status"]),
            final_status=RunStatus(value["final_status"]),
            final_answer=value.get("final_answer"),
            is_correct=value.get("is_correct"),
            recovered=bool(value.get("recovered", False)),
        )


def _budget_key(case: EvalCase) -> str:
    return {
        TaskType.MULTIPLE_CHOICE: "multiple_choice",
        TaskType.INSTRUCTION_FOLLOWING: "ifeval",
        TaskType.RUBRIC: "product_rubric",
    }[case.task_type]


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _http_status(value: Any) -> int | None:
    direct = _attribute(value, "status_code")
    if isinstance(direct, int):
        return direct
    wrapped = _attribute(value, "_response")
    status = _attribute(wrapped, "status_code")
    return status if isinstance(status, int) else None


def _error_evidence(exc: BaseException) -> tuple[str, str | None, int | None]:
    status = _http_status(exc)
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "timeout", None, status
    if status == 429:
        return "rate_limit", "429", status
    if status in (401, 403):
        return "authentication", str(status), status
    if status is not None and status >= 500:
        return "server", str(status), status
    return "request", str(status) if status is not None else None, status


def _safe_retry_headers(exc: BaseException) -> dict[str, str]:
    response = _attribute(exc, "response")
    headers = _attribute(response, "headers") or _attribute(exc, "headers")
    if not isinstance(headers, Mapping):
        return {}
    allowed = {"retry-after", "x-ratelimit-reset"}
    return {
        str(key): str(value)
        for key, value in headers.items()
        if str(key).lower() in allowed
    }


def call_once(
    config: ModelConfig,
    case: EvalCase,
    client_factory: ClientFactory,
    clock: Clock = time.monotonic,
    *,
    max_tokens: int | None = None,
    attempt_number: int = 1,
    wait_seconds: float = 0.0,
) -> AttemptRecord:
    """Make one request and preserve enough evidence to audit its outcome."""

    budget = config.output_budgets_by_task_type[_budget_key(case)]
    token_limit = budget.initial if max_tokens is None else max_tokens
    start = clock()
    try:
        client = client_factory(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
        )
        request: dict[str, Any] = {
            "model": config.model,
            "messages": build_prompt(case),
            "max_tokens": token_limit,
            "stream": False,
        }
        if config.name in {"KIMI", "DEEPSEEK"}:
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        elif config.name == "GLM":
            request["temperature"] = 0
        response = client.chat.completions.create(**request)
        end = clock()

        choices = _attribute(response, "choices", []) or []
        choice = choices[0] if choices else None
        finish_reason = _attribute(choice, "finish_reason")
        message = _attribute(choice, "message")
        content = _attribute(message, "content")
        raw_response = content if isinstance(content, str) else None
        usage = _attribute(response, "usage")
        common = {
            "model_name": config.name,
            "model_id": config.model,
            "case_id": case.id,
            "attempt_number": attempt_number,
            "max_tokens": token_limit,
            "raw_response": raw_response,
            "finish_reason": finish_reason,
            "prompt_tokens": _attribute(usage, "prompt_tokens"),
            "completion_tokens": _attribute(usage, "completion_tokens"),
            "total_tokens": _attribute(usage, "total_tokens"),
            "latency_ms": (end - start) * 1000,
            "http_status": _http_status(response),
            "wait_seconds": wait_seconds,
        }

        if finish_reason == "length":
            return AttemptRecord(
                **common,
                status=RunStatus.TRUNCATED,
                normalized_answer=None,
                is_correct=None,
                error_type="truncation",
                error_message="provider reported finish_reason=length",
            )
        if not finish_reason:
            return AttemptRecord(
                **common,
                status=RunStatus.API_ERROR,
                normalized_answer=None,
                is_correct=None,
                error_type="invalid_response",
                error_message="response did not include finish_reason",
            )
        if raw_response is None or not raw_response.strip():
            return AttemptRecord(
                **common,
                status=RunStatus.API_ERROR,
                normalized_answer=None,
                is_correct=None,
                error_type="empty_response",
                error_message="response content was empty",
            )
        try:
            if case.task_type is TaskType.MULTIPLE_CHOICE:
                grade = grade_multiple_choice(case, raw_response)
            elif case.task_type is TaskType.INSTRUCTION_FOLLOWING:
                grade = grade_ifeval(case, raw_response)
            else:
                grade = grade_rubric(case, raw_response)
        except (RuntimeError, ValueError) as exc:
            return AttemptRecord(
                **common,
                status=RunStatus.GRADER_ERROR,
                normalized_answer=None,
                is_correct=None,
                error_type="grader",
                error_message=str(exc),
            )
        return AttemptRecord(
            **common,
            status=grade.status,
            normalized_answer=grade.normalized_answer,
            is_correct=grade.is_correct,
            error_type=("format" if grade.status is RunStatus.FORMAT_ERROR else None),
            error_message=grade.reason,
            grader_details=dict(grade.details),
        )
    except Exception as exc:
        end = clock()
        error_type, error_code, status = _error_evidence(exc)
        return AttemptRecord(
            model_name=config.name,
            model_id=config.model,
            case_id=case.id,
            attempt_number=attempt_number,
            max_tokens=token_limit,
            status=RunStatus.API_ERROR,
            raw_response=None,
            normalized_answer=None,
            is_correct=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=(end - start) * 1000,
            http_status=status,
            error_type=error_type,
            error_code=error_code,
            error_message="provider request failed; inspect error_type and status",
            wait_seconds=wait_seconds,
            retry_headers=_safe_retry_headers(exc),
        )


def run_case_with_recovery(
    config: ModelConfig,
    case: EvalCase,
    client_factory: ClientFactory,
    clock: Clock = time.monotonic,
    *,
    initial_wait_seconds: float = 0.0,
    recovery_wait_fn: Callable[[], float] | None = None,
) -> CaseRunResult:
    """Retry a truncated response once with the task's recovery budget."""

    budget = config.output_budgets_by_task_type[_budget_key(case)]
    attempts = [
        call_once(
            config,
            case,
            client_factory,
            clock,
            max_tokens=budget.initial,
            attempt_number=1,
            wait_seconds=initial_wait_seconds,
        )
    ]
    if attempts[0].status is RunStatus.TRUNCATED:
        recovery_wait = recovery_wait_fn() if recovery_wait_fn else 0.0
        attempts.append(
            call_once(
                config,
                case,
                client_factory,
                clock,
                max_tokens=budget.recovery,
                attempt_number=2,
                wait_seconds=recovery_wait,
            )
        )

    final = attempts[-1]
    return CaseRunResult(
        model_name=config.name,
        model_id=config.model,
        case_id=case.id,
        task_type=case.task_type,
        subject=case.subject,
        attempts=tuple(attempts),
        first_pass_status=attempts[0].status,
        final_status=final.status,
        final_answer=final.normalized_answer,
        is_correct=final.is_correct,
        recovered=(
            attempts[0].status is RunStatus.TRUNCATED
            and final.status is not RunStatus.TRUNCATED
        ),
    )
