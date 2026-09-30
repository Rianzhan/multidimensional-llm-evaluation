import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_eval.config import ModelConfig
from llm_eval.providers import AttemptRecord, CaseRunResult
from llm_eval.schema import EvalCase, RunStatus, TaskType


def recovery_module():
    try:
        return importlib.import_module("llm_eval.recovery")
    except ModuleNotFoundError:
        pytest.fail("llm_eval.recovery is not implemented")


def make_result(model: str, case_id: str, status: RunStatus) -> CaseRunResult:
    answer = "A" if status is RunStatus.CORRECT else "B" if status is RunStatus.WRONG else None
    is_correct = True if status is RunStatus.CORRECT else False if status is RunStatus.WRONG else None
    attempt = AttemptRecord(
        model_name=model,
        model_id=f"{model.lower()}-model",
        case_id=case_id,
        attempt_number=1,
        max_tokens=256,
        status=status,
        raw_response=answer,
        normalized_answer=answer,
        is_correct=is_correct,
        finish_reason="length" if status is RunStatus.TRUNCATED else "stop",
        prompt_tokens=10,
        completion_tokens=256,
        total_tokens=266,
        latency_ms=100,
        http_status=200,
        error_type="truncation" if status is RunStatus.TRUNCATED else None,
    )
    return CaseRunResult(
        model_name=model,
        model_id=f"{model.lower()}-model",
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


def write_checkpoint(path: Path, *results: CaseRunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(result.to_dict()) for result in results) + "\n",
        encoding="utf-8",
    )


def case_payload(case_id: str, task_type: str = "multiple_choice") -> dict:
    is_choice = task_type == "multiple_choice"
    return {
        "id": case_id,
        "task_type": task_type,
        "subject": "math" if is_choice else "instruction_following",
        "difficulty": "hard",
        "source_dataset": "fixture",
        "source_id": case_id,
        "source_split": "test",
        "source_url": "https://example.test",
        "prompt": "Question?",
        "choices": ["a", "b", "c", "d"] if is_choice else [],
        "reference_answer": "A" if is_choice else None,
        "grader": {"type": "strict_choice" if is_choice else "official_ifeval"},
        "derived_from": None,
        "metadata": {},
    }


def write_jsonl(path: Path, *payloads: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(payload) for payload in payloads) + "\n",
        encoding="utf-8",
    )


def test_collect_recovery_targets_selects_only_final_truncations(tmp_path):
    write_checkpoint(
        tmp_path / "base_checkpoint.jsonl",
        make_result("GLM", "base-truncated", RunStatus.TRUNCATED),
        make_result("KIMI", "base-correct", RunStatus.CORRECT),
    )
    write_checkpoint(
        tmp_path / "stability_checkpoint_1.jsonl",
        make_result("DEEPSEEK", "stable::run-1", RunStatus.TRUNCATED),
    )
    write_checkpoint(
        tmp_path / "robustness_checkpoint.jsonl",
        make_result("KIMI", "robust-truncated", RunStatus.TRUNCATED),
    )

    targets = recovery_module().collect_recovery_targets(tmp_path)
    source_results = recovery_module().collect_source_results(tmp_path)

    assert [
        (target.suite, target.result.model_name, target.result.case_id)
        for target in targets
    ] == [
        ("base", "GLM", "base-truncated"),
        ("stability-1", "DEEPSEEK", "stable::run-1"),
        ("robustness", "KIMI", "robust-truncated"),
    ]
    assert len(source_results) == 4
    assert any(
        item.result.case_id == "base-correct"
        and item.result.final_status is RunStatus.CORRECT
        for item in source_results
    )


def test_build_case_catalog_reconstructs_stability_repeat_ids(tmp_path):
    write_jsonl(tmp_path / "base_90.jsonl", case_payload("base-case"))
    write_jsonl(tmp_path / "robustness_30.jsonl", case_payload("robust-case"))
    write_jsonl(tmp_path / "stability_15.jsonl", case_payload("stable-case"))

    catalog = recovery_module().build_case_catalog(tmp_path, stability_runs=3)

    assert set(catalog) == {
        "base-case",
        "robust-case",
        "stable-case::run-1",
        "stable-case::run-2",
        "stable-case::run-3",
    }
    assert catalog["stable-case::run-2"].derived_from == "stable-case"
    assert catalog["stable-case::run-2"].metadata["stability_run"] == 2


def test_extended_budget_config_uses_declared_recovery_limits():
    config = ModelConfig(
        name="KIMI",
        api_key="test-value",
        model="kimi-test",
        base_url="https://example.test/v1",
    )

    expanded = recovery_module().with_extended_budgets(config)

    assert expanded.output_budgets_by_task_type["multiple_choice"].initial == 4096
    assert expanded.output_budgets_by_task_type["multiple_choice"].recovery == 4096
    assert expanded.output_budgets_by_task_type["ifeval"].initial == 8192
    assert expanded.output_budgets_by_task_type["ifeval"].recovery == 8192
    assert expanded.output_budgets_by_task_type["product_rubric"] == (
        config.output_budgets_by_task_type["product_rubric"]
    )


def test_extended_budget_config_accepts_a_larger_final_retry_limit():
    config = ModelConfig(
        name="KIMI",
        api_key="test-value",
        model="kimi-test",
        base_url="https://example.test/v1",
    )

    expanded = recovery_module().with_extended_budgets(
        config,
        multiple_choice_tokens=8192,
    )

    assert expanded.output_budgets_by_task_type["multiple_choice"].initial == 8192
    assert expanded.output_budgets_by_task_type["multiple_choice"].recovery == 8192


def test_extended_case_uses_one_declared_budget_attempt():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content="unfinished"),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=4096,
                    total_tokens=4106,
                ),
                _response=SimpleNamespace(status_code=200),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    config = recovery_module().with_extended_budgets(
        ModelConfig(
            name="KIMI",
            api_key="test-value",
            model="kimi-test",
            base_url="https://example.test/v1",
        )
    )
    case = EvalCase.from_dict(case_payload("case-1"))

    result = recovery_module().run_extended_case(
        config,
        case,
        client_factory=lambda **_: client,
        clock=iter((1.0, 2.0)).__next__,
    )

    assert len(result.attempts) == 1
    assert result.attempts[0].max_tokens == 4096
    assert result.final_status is RunStatus.TRUNCATED
    assert result.recovered is False
    assert calls[0]["max_tokens"] == 4096


def test_recovery_experiment_runs_only_sparse_model_case_pairs(tmp_path):
    module = recovery_module()
    glm_result = make_result("GLM", "case-glm", RunStatus.TRUNCATED)
    kimi_result = make_result("KIMI", "case-kimi", RunStatus.TRUNCATED)
    targets = [
        module.RecoveryTarget("base", glm_result),
        module.RecoveryTarget("robustness", kimi_result),
    ]
    catalog = {
        "case-glm": EvalCase.from_dict(case_payload("case-glm")),
        "case-kimi": EvalCase.from_dict(case_payload("case-kimi")),
    }
    configs = [
        ModelConfig("GLM", "key", "glm", "https://example.test/v1"),
        ModelConfig("KIMI", "key", "kimi", "https://example.test/v1"),
    ]
    calls = []

    def fake_runner(cases, selected_configs, checkpoint, **kwargs):
        case_list = list(cases)
        config_list = list(selected_configs)
        calls.append(
            (
                [case.id for case in case_list],
                [config.name for config in config_list],
                checkpoint.name,
                config_list[0].output_budgets_by_task_type[
                    "multiple_choice"
                ].initial,
                kwargs["run_case_fn"],
            )
        )
        source = glm_result if config_list[0].name == "GLM" else kimi_result
        return [source]

    results = module.run_recovery_experiment(
        targets,
        catalog,
        configs,
        tmp_path,
        runner_fn=fake_runner,
    )

    assert [(call[0], call[1]) for call in calls] == [
        (["case-glm"], ["GLM"]),
        (["case-kimi"], ["KIMI"]),
    ]
    assert [call[2] for call in calls] == [
        "recovery_glm_checkpoint.jsonl",
        "recovery_kimi_checkpoint.jsonl",
    ]
    assert all(call[3] == 4096 for call in calls)
    assert all(call[4] is module.run_extended_case for call in calls)
    assert results == [glm_result, kimi_result]


def test_recovery_experiment_forwards_final_retry_budget(tmp_path):
    module = recovery_module()
    truncated = make_result("GLM", "case-glm", RunStatus.TRUNCATED)
    target = module.RecoveryTarget("final-retry", truncated)
    catalog = {"case-glm": EvalCase.from_dict(case_payload("case-glm"))}
    config = ModelConfig("GLM", "key", "glm", "https://example.test/v1")
    observed_budgets = []

    def fake_runner(cases, selected_configs, checkpoint, **kwargs):
        del cases, checkpoint, kwargs
        selected = list(selected_configs)
        observed_budgets.append(
            selected[0].output_budgets_by_task_type["multiple_choice"].initial
        )
        return [truncated]

    module.run_recovery_experiment(
        [target],
        catalog,
        [config],
        tmp_path,
        runner_fn=fake_runner,
        multiple_choice_tokens=8192,
    )

    assert observed_budgets == [8192]


def test_prior_recovery_loader_selects_remaining_targets_and_merges_retry(tmp_path):
    module = recovery_module()
    source_run = tmp_path / "formal-source"
    prior_dir = tmp_path / "recovery-source"
    prior_dir.mkdir()
    glm_truncated = make_result("GLM", "case-glm", RunStatus.TRUNCATED)
    kimi_format_error = make_result("KIMI", "case-kimi", RunStatus.FORMAT_ERROR)
    payload = {
        "schema_version": 1,
        "experiment_type": "post_hoc_truncation_recovery",
        "source_run": str(source_run),
        "budget_policy": {
            "multiple_choice": 4096,
            "instruction_following": 8192,
        },
        "results": [glm_truncated.to_dict(), kimi_format_error.to_dict()],
    }
    (prior_dir / "recovery_results.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    loaded_source, prior_results = module.load_recovery_run(prior_dir)
    prior_budget = module.load_recovery_budget_policy(prior_dir)
    targets = module.collect_remaining_recovery_targets(prior_results)
    merged = module.merge_recovery_results(
        prior_results,
        [make_result("GLM", "case-glm", RunStatus.CORRECT)],
    )

    assert loaded_source == source_run.resolve()
    assert prior_budget == {
        "multiple_choice": 4096,
        "instruction_following": 8192,
    }
    assert [(item.result.model_name, item.result.case_id) for item in targets] == [
        ("GLM", "case-glm")
    ]
    assert {
        (result.model_name, result.case_id): result.final_status for result in merged
    } == {
        ("GLM", "case-glm"): RunStatus.CORRECT,
        ("KIMI", "case-kimi"): RunStatus.FORMAT_ERROR,
    }


def test_recovery_summary_keeps_strict_scores_and_reports_recovered_scores():
    module = recovery_module()
    source_results = [
        module.RecoveryTarget(
            "base", make_result("GLM", "glm-correct", RunStatus.CORRECT)
        ),
        module.RecoveryTarget(
            "base", make_result("GLM", "glm-truncated", RunStatus.TRUNCATED)
        ),
        module.RecoveryTarget(
            "robustness",
            make_result("KIMI", "kimi-truncated", RunStatus.TRUNCATED),
        ),
    ]
    recovered_results = [
        make_result("GLM", "glm-truncated", RunStatus.CORRECT),
        make_result("KIMI", "kimi-truncated", RunStatus.FORMAT_ERROR),
    ]

    summary = module.build_recovery_summary(source_results, recovered_results)

    glm = summary["by_model"]["GLM"]
    assert glm["strict_correct_count"] == 1
    assert glm["effective_correct_count"] == 2
    assert glm["strict_end_to_end_accuracy"] == 0.5
    assert glm["effective_end_to_end_accuracy"] == 1.0
    assert glm["accuracy_delta_percentage_points"] == 50.0
    assert glm["truncation_recovery_rate"] == 1.0

    kimi = summary["by_model"]["KIMI"]
    assert kimi["strict_correct_count"] == 0
    assert kimi["effective_correct_count"] == 0
    assert kimi["recovery_format_error_count"] == 1
    assert kimi["truncation_recovery_rate"] == 1.0
    assert summary["target_count"] == 2
    assert summary["completed_target_count"] == 2


def test_write_recovery_outputs_creates_separate_auditable_report(tmp_path):
    module = recovery_module()
    source_run = tmp_path / "formal-source"
    output_dir = tmp_path / "recovery-output"
    source_results = [
        module.RecoveryTarget(
            "base", make_result("GLM", "case-1", RunStatus.TRUNCATED)
        )
    ]
    recovered_results = [make_result("GLM", "case-1", RunStatus.CORRECT)]

    artifacts = module.write_recovery_outputs(
        output_dir,
        source_run,
        source_results,
        recovered_results,
    )

    for name in (
        "recovery_results.json",
        "recovery_summary.json",
        "recovery_report.md",
    ):
        assert (output_dir / name).exists()
    payload = json.loads(
        (output_dir / "recovery_results.json").read_text(encoding="utf-8")
    )
    assert payload["source_run"] == str(source_run.resolve())
    assert payload["results"][0]["final_status"] == "correct"
    report = (output_dir / "recovery_report.md").read_text(encoding="utf-8")
    assert "后验恢复实验" in report
    assert "严格端到端正确率" in report
    assert "恢复后端到端正确率" in report
    assert artifacts["summary"]["target_count"] == 1


def test_write_recovery_outputs_records_the_actual_final_retry_budget(tmp_path):
    module = recovery_module()
    source_run = tmp_path / "formal-source"
    output_dir = tmp_path / "recovery-output"
    source_results = [
        module.RecoveryTarget(
            "base", make_result("GLM", "case-1", RunStatus.TRUNCATED)
        )
    ]

    module.write_recovery_outputs(
        output_dir,
        source_run,
        source_results,
        [make_result("GLM", "case-1", RunStatus.CORRECT)],
        budget_policy={"multiple_choice": 8192, "instruction_following": 8192},
    )

    payload = json.loads(
        (output_dir / "recovery_results.json").read_text(encoding="utf-8")
    )
    assert payload["budget_policy"] == {
        "multiple_choice": 8192,
        "instruction_following": 8192,
    }


def test_write_recovery_outputs_explains_a_two_stage_budget_policy(tmp_path):
    module = recovery_module()
    output_dir = tmp_path / "recovery-output"
    source_results = [
        module.RecoveryTarget(
            "base", make_result("GLM", "case-1", RunStatus.TRUNCATED)
        )
    ]

    module.write_recovery_outputs(
        output_dir,
        tmp_path / "formal-source",
        source_results,
        [make_result("GLM", "case-1", RunStatus.CORRECT)],
        budget_policy={
            "multiple_choice": 4096,
            "instruction_following": 8192,
            "final_retry_multiple_choice": 8192,
        },
    )

    report = (output_dir / "recovery_report.md").read_text(encoding="utf-8")
    assert "第一轮扩展预算：选择题 4096 tokens" in report
    assert "最终重试选择题预算：8192 tokens" in report
