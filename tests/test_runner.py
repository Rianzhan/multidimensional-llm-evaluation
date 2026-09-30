import json
import threading
from pathlib import Path

from llm_eval.config import ModelConfig
from llm_eval.providers import AttemptRecord, CaseRunResult
from llm_eval.runner import evaluate_suite
from llm_eval.rate_limit import ProviderRateLimiter
from llm_eval.schema import EvalCase, RunStatus, TaskType


def make_case(case_id):
    return EvalCase(
        id=case_id,
        task_type=TaskType.MULTIPLE_CHOICE,
        subject="math",
        difficulty="hard",
        source_dataset="fixture",
        source_id=case_id,
        source_split="test",
        source_url="https://example.test",
        prompt="Question?",
        choices=("a", "b", "c", "d"),
        reference_answer="A",
        grader={"type": "strict_choice"},
    )


def make_config(name):
    return ModelConfig(
        name=name,
        api_key="test-value",
        model=f"{name.lower()}-model",
        base_url="https://example.test/v1",
        min_request_interval_seconds=0,
    )


def make_result(config, case, status=RunStatus.CORRECT):
    correct = True if status is RunStatus.CORRECT else False if status is RunStatus.WRONG else None
    answer = "A" if status is RunStatus.CORRECT else "B" if status is RunStatus.WRONG else None
    attempt = AttemptRecord(
        model_name=config.name,
        model_id=config.model,
        case_id=case.id,
        attempt_number=1,
        max_tokens=64,
        status=status,
        raw_response=answer,
        normalized_answer=answer,
        is_correct=correct,
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=1,
        total_tokens=11,
        latency_ms=1,
        http_status=200,
    )
    return CaseRunResult(
        model_name=config.name,
        model_id=config.model,
        case_id=case.id,
        task_type=case.task_type,
        subject=case.subject,
        attempts=(attempt,),
        first_pass_status=status,
        final_status=status,
        final_answer=answer,
        is_correct=correct,
        recovered=False,
    )


def test_provider_queues_start_in_parallel_and_results_have_stable_order(tmp_path):
    cases = [make_case("case-2"), make_case("case-1")]
    configs = [make_config("GLM"), make_config("KIMI"), make_config("DEEPSEEK")]
    barrier = threading.Barrier(3, timeout=2)
    first_seen = set()
    lock = threading.Lock()

    def run_case(config, case, **_):
        with lock:
            first_for_provider = config.name not in first_seen
            first_seen.add(config.name)
        if first_for_provider:
            barrier.wait()
        return make_result(config, case)

    results = evaluate_suite(
        cases,
        configs,
        tmp_path / "checkpoint.jsonl",
        run_case_fn=run_case,
    )

    assert [(item.case_id, item.model_name) for item in results] == [
        ("case-2", "GLM"),
        ("case-2", "KIMI"),
        ("case-2", "DEEPSEEK"),
        ("case-1", "GLM"),
        ("case-1", "KIMI"),
        ("case-1", "DEEPSEEK"),
    ]


def test_resume_skips_complete_pairs_and_runs_only_missing_pairs(tmp_path):
    cases = [make_case("case-1"), make_case("case-2")]
    configs = [make_config("GLM"), make_config("KIMI")]
    checkpoint = tmp_path / "checkpoint.jsonl"
    existing = make_result(configs[0], cases[0])
    checkpoint.write_text(json.dumps(existing.to_dict()) + "\n", encoding="utf-8")
    calls = []

    def run_case(config, case, **_):
        calls.append((config.name, case.id))
        return make_result(config, case)

    results = evaluate_suite(
        cases, configs, checkpoint, run_case_fn=run_case
    )

    assert ("GLM", "case-1") not in calls
    assert len(calls) == 3
    assert len(results) == 4


def test_duplicate_checkpoint_rows_use_latest_complete_record(tmp_path):
    case = make_case("case-1")
    config = make_config("GLM")
    checkpoint = tmp_path / "checkpoint.jsonl"
    first = make_result(config, case, RunStatus.WRONG)
    latest = make_result(config, case, RunStatus.CORRECT)
    checkpoint.write_text(
        json.dumps(first.to_dict()) + "\n" + json.dumps(latest.to_dict()) + "\n",
        encoding="utf-8",
    )

    results = evaluate_suite(
        [case],
        [config],
        checkpoint,
        run_case_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed pair should not be requested again")
        ),
    )

    assert len(results) == 1
    assert results[0].final_status is RunStatus.CORRECT


def test_invalid_or_partial_checkpoint_line_does_not_hide_pending_work(tmp_path):
    case = make_case("case-1")
    config = make_config("GLM")
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text('{"model_name":"GLM"', encoding="utf-8")
    calls = []

    results = evaluate_suite(
        [case],
        [config],
        checkpoint,
        run_case_fn=lambda cfg, item, **_: calls.append((cfg.name, item.id))
        or make_result(cfg, item),
    )

    assert calls == [("GLM", "case-1")]
    assert results[0].final_status is RunStatus.CORRECT


def test_shared_limiter_preserves_kimi_interval_across_suite_phases(tmp_path):
    fake_now = [0.0]
    sleeps = []

    def clock():
        return fake_now[0]

    def sleep(seconds):
        sleeps.append(seconds)
        fake_now[0] += seconds

    config = make_config("KIMI")
    config = ModelConfig(
        **{
            **config.__dict__,
            "min_request_interval_seconds": 22,
        }
    )
    shared = {
        "KIMI": ProviderRateLimiter(22, clock=clock, sleep_fn=sleep)
    }
    run_case = lambda cfg, item, **_: make_result(cfg, item)

    evaluate_suite(
        [make_case("base")],
        [config],
        tmp_path / "base.jsonl",
        run_case_fn=run_case,
        clock=clock,
        sleep_fn=sleep,
        limiters=shared,
    )
    evaluate_suite(
        [make_case("stability")],
        [config],
        tmp_path / "stability.jsonl",
        run_case_fn=run_case,
        clock=clock,
        sleep_fn=sleep,
        limiters=shared,
    )

    assert sleeps == [22]
