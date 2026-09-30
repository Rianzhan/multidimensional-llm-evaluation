from types import SimpleNamespace

import pytest

from llm_eval.config import ModelConfig
from llm_eval.providers import call_once, run_case_with_recovery
from llm_eval.schema import EvalCase, RunStatus, TaskType


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def response(content="A", finish_reason="stop", status_code=200):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, reasoning_content="B"),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=1, total_tokens=21),
        _response=SimpleNamespace(status_code=status_code),
    )


def model_config(name="GLM"):
    return ModelConfig(
        name=name,
        api_key="test-value",
        model="test-model",
        base_url="https://example.test/v1",
    )


def case(reference="A"):
    return EvalCase(
        id="case-1",
        task_type=TaskType.MULTIPLE_CHOICE,
        subject="math",
        difficulty="hard",
        source_dataset="fixture",
        source_id="1",
        source_split="test",
        source_url="https://example.test/source",
        prompt="Question?",
        choices=("a", "b", "c", "d"),
        reference_answer=reference,
        grader={"type": "strict_choice"},
    )


def clock_with(*values):
    iterator = iter(values)
    return lambda: next(iterator)


def test_call_once_captures_response_evidence_and_uses_64_tokens():
    completions = FakeCompletions([response(content="A")])

    attempt = call_once(
        model_config(),
        case(),
        client_factory=lambda **_: FakeClient(completions),
        clock=clock_with(10.0, 10.125),
    )

    assert attempt.status is RunStatus.CORRECT
    assert attempt.raw_response == "A"
    assert attempt.finish_reason == "stop"
    assert attempt.prompt_tokens == 20
    assert attempt.completion_tokens == 1
    assert attempt.total_tokens == 21
    assert attempt.latency_ms == pytest.approx(125.0)
    assert attempt.http_status == 200
    assert completions.calls[0]["max_tokens"] == 64


def test_legal_wrong_answer_is_model_error():
    completions = FakeCompletions([response(content="B")])
    attempt = call_once(
        model_config(), case(), lambda **_: FakeClient(completions), clock_with(1, 2)
    )

    assert attempt.status is RunStatus.WRONG
    assert attempt.is_correct is False


def test_invalid_text_is_format_error():
    completions = FakeCompletions([response(content="The answer is A")])
    attempt = call_once(
        model_config(), case(), lambda **_: FakeClient(completions), clock_with(1, 2)
    )

    assert attempt.status is RunStatus.FORMAT_ERROR
    assert attempt.is_correct is None


def test_length_finish_reason_is_truncation_even_if_content_has_a_letter():
    completions = FakeCompletions([response(content="A", finish_reason="length")])
    attempt = call_once(
        model_config(), case(), lambda **_: FakeClient(completions), clock_with(1, 2)
    )

    assert attempt.status is RunStatus.TRUNCATED
    assert attempt.finish_reason == "length"


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [(None, "stop"), ("", "stop"), (None, None)],
)
def test_empty_content_or_missing_finish_reason_is_not_wrong(content, finish_reason):
    completions = FakeCompletions(
        [response(content=content, finish_reason=finish_reason)]
    )
    attempt = call_once(
        model_config(), case(), lambda **_: FakeClient(completions), clock_with(1, 2)
    )

    assert attempt.status is RunStatus.API_ERROR
    assert attempt.is_correct is None


def test_reasoning_content_is_not_used_as_the_answer():
    completions = FakeCompletions([response(content=None, finish_reason="stop")])
    attempt = call_once(
        model_config(), case(), lambda **_: FakeClient(completions), clock_with(1, 2)
    )

    assert attempt.raw_response is None
    assert attempt.status is RunStatus.API_ERROR


def test_exception_is_redacted_and_classified_as_api_error():
    completions = FakeCompletions([TimeoutError("secret-value in network error")])
    attempt = call_once(
        model_config(), case(), lambda **_: FakeClient(completions), clock_with(1, 2)
    )

    assert attempt.status is RunStatus.API_ERROR
    assert attempt.error_type == "timeout"
    assert "secret-value" not in (attempt.error_message or "")


def test_truncation_recovery_preserves_both_attempts_and_raises_budget():
    completions = FakeCompletions(
        [
            response(content="A", finish_reason="length"),
            response(content="A", finish_reason="stop"),
        ]
    )
    times = clock_with(1, 2, 3, 4)

    result = run_case_with_recovery(
        model_config(),
        case(),
        client_factory=lambda **_: FakeClient(completions),
        clock=times,
    )

    assert len(result.attempts) == 2
    assert result.first_pass_status is RunStatus.TRUNCATED
    assert result.final_status is RunStatus.CORRECT
    assert result.recovered is True
    assert [call["max_tokens"] for call in completions.calls] == [64, 256]


def test_truncation_recovery_records_limiter_wait_for_each_request():
    completions = FakeCompletions(
        [
            response(content="A", finish_reason="length"),
            response(content="A", finish_reason="stop"),
        ]
    )

    result = run_case_with_recovery(
        model_config("KIMI"),
        case(),
        client_factory=lambda **_: FakeClient(completions),
        clock=clock_with(1, 2, 3, 4),
        initial_wait_seconds=3,
        recovery_wait_fn=lambda: 22,
    )

    assert [attempt.wait_seconds for attempt in result.attempts] == [3, 22]
