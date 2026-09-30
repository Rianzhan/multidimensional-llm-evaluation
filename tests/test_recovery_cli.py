import importlib
import json
from pathlib import Path

import pytest

from llm_eval.providers import AttemptRecord, CaseRunResult
from llm_eval.schema import RunStatus, TaskType


def recovery_cli_module():
    try:
        return importlib.import_module("llm_eval.recovery_cli")
    except ModuleNotFoundError:
        pytest.fail("llm_eval.recovery_cli is not implemented")


def truncated_result() -> CaseRunResult:
    attempt = AttemptRecord(
        model_name="GLM",
        model_id="glm-5.2",
        case_id="base-case",
        attempt_number=1,
        max_tokens=256,
        status=RunStatus.TRUNCATED,
        raw_response="unfinished",
        normalized_answer=None,
        is_correct=None,
        finish_reason="length",
        prompt_tokens=10,
        completion_tokens=256,
        total_tokens=266,
        latency_ms=100,
        http_status=200,
        error_type="truncation",
    )
    return CaseRunResult(
        model_name="GLM",
        model_id="glm-5.2",
        case_id="base-case",
        task_type=TaskType.MULTIPLE_CHOICE,
        subject="math",
        attempts=(attempt,),
        first_pass_status=RunStatus.TRUNCATED,
        final_status=RunStatus.TRUNCATED,
        final_answer=None,
        is_correct=None,
        recovered=False,
    )


def write_fixture_data(data_dir: Path) -> None:
    choice = {
        "id": "base-case",
        "task_type": "multiple_choice",
        "subject": "math",
        "difficulty": "hard",
        "source_dataset": "fixture",
        "source_id": "base-case",
        "source_split": "test",
        "source_url": "https://example.test",
        "prompt": "Question?",
        "choices": ["a", "b", "c", "d"],
        "reference_answer": "A",
        "grader": {"type": "strict_choice"},
        "derived_from": None,
        "metadata": {},
    }
    data_dir.mkdir(parents=True)
    (data_dir / "base_90.jsonl").write_text(
        json.dumps(choice) + "\n", encoding="utf-8"
    )
    robust = {**choice, "id": "robust-case", "source_id": "robust-case"}
    (data_dir / "robustness_30.jsonl").write_text(
        json.dumps(robust) + "\n", encoding="utf-8"
    )
    stable = {**choice, "id": "stable-case", "source_id": "stable-case"}
    (data_dir / "stability_15.jsonl").write_text(
        json.dumps(stable) + "\n", encoding="utf-8"
    )


def test_recovery_cli_dry_run_reports_sparse_request_count(tmp_path, capsys):
    source_run = tmp_path / "source"
    source_run.mkdir()
    result = truncated_result()
    (source_run / "base_checkpoint.jsonl").write_text(
        json.dumps(result.to_dict()) + "\n", encoding="utf-8"
    )
    data_dir = tmp_path / "data"
    write_fixture_data(data_dir)

    exit_code = recovery_cli_module().main(
        [
            "--source-run",
            str(source_run),
            "--output-dir",
            str(tmp_path / "output"),
            "--data-dir",
            str(data_dir),
            "--dry-run",
        ],
        environ={"GLM_API_KEY": "test-value"},
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Recovery targets: 1" in output
    assert "GLM: 1" in output
    assert not (tmp_path / "output").exists()


def test_recovery_cli_refuses_existing_checkpoint_without_resume(tmp_path, capsys):
    source_run = tmp_path / "source"
    source_run.mkdir()
    result = truncated_result()
    (source_run / "base_checkpoint.jsonl").write_text(
        json.dumps(result.to_dict()) + "\n", encoding="utf-8"
    )
    data_dir = tmp_path / "data"
    write_fixture_data(data_dir)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "recovery_glm_checkpoint.jsonl").write_text(
        json.dumps(result.to_dict()) + "\n", encoding="utf-8"
    )

    exit_code = recovery_cli_module().main(
        [
            "--source-run",
            str(source_run),
            "--output-dir",
            str(output_dir),
            "--data-dir",
            str(data_dir),
        ],
        environ={"GLM_API_KEY": "test-value"},
    )

    error = capsys.readouterr().err
    assert exit_code == 2
    assert "--resume" in error


def test_recovery_cli_dry_run_can_retry_only_remaining_prior_truncations(
    tmp_path, capsys
):
    source_run = tmp_path / "source"
    source_run.mkdir()
    result = truncated_result()
    (source_run / "base_checkpoint.jsonl").write_text(
        json.dumps(result.to_dict()) + "\n", encoding="utf-8"
    )
    prior_run = tmp_path / "prior-recovery"
    prior_run.mkdir()
    (prior_run / "recovery_results.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_type": "post_hoc_truncation_recovery",
                "source_run": str(source_run.resolve()),
                "budget_policy": {
                    "multiple_choice": 4096,
                    "instruction_following": 8192,
                },
                "results": [result.to_dict()],
            }
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    write_fixture_data(data_dir)

    exit_code = recovery_cli_module().main(
        [
            "--source-run",
            str(source_run),
            "--prior-recovery-run",
            str(prior_run),
            "--multiple-choice-tokens",
            "8192",
            "--output-dir",
            str(tmp_path / "output"),
            "--data-dir",
            str(data_dir),
            "--dry-run",
        ],
        environ={"GLM_API_KEY": "test-value"},
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Recovery targets: 1" in output
    assert "GLM: 1" in output
    assert "Multiple-choice budget: 8192" in output
