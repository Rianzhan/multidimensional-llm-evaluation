from pathlib import Path

from llm_eval.cli import main, parse_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def environment():
    return {
        "GLM_API_KEY": "glm-test",
        "KIMI_API_KEY": "kimi-test",
        "DEEPSEEK_API_KEY": "deepseek-test",
    }


def test_parse_args_exposes_v2_controls():
    args = parse_args(
        [
            "--dry-run",
            "--resume",
            "--models",
            "GLM",
            "KIMI",
            "--dataset",
            "custom.jsonl",
            "--stability-dataset",
            "stable.jsonl",
            "--stability-runs",
            "5",
        ]
    )

    assert args.dry_run is True
    assert args.resume is True
    assert args.models == ["GLM", "KIMI"]
    assert args.dataset == Path("custom.jsonl")
    assert args.stability_dataset == Path("stable.jsonl")
    assert args.stability_runs == 5


def test_dry_run_validates_and_estimates_without_calling_models(capsys):
    def forbidden_runner(*_args, **_kwargs):
        raise AssertionError("dry-run must not call providers")

    exit_code = main(
        [
            "--dry-run",
            "--dataset",
            str(PROJECT_ROOT / "data" / "base_90.jsonl"),
            "--stability-dataset",
            str(PROJECT_ROOT / "data" / "stability_15.jsonl"),
        ],
        environ=environment(),
        runner_fn=forbidden_runner,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "基础题: 90" in output
    assert "基础请求: 270" in output
    assert "稳定性请求: 135" in output
    assert "Kimi 最短预计" in output


def test_dry_run_model_filter_changes_request_count(capsys):
    exit_code = main(
        [
            "--dry-run",
            "--models",
            "GLM",
            "DEEPSEEK",
            "--dataset",
            str(PROJECT_ROOT / "data" / "base_90.jsonl"),
            "--stability-dataset",
            str(PROJECT_ROOT / "data" / "stability_15.jsonl"),
        ],
        environ=environment(),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "基础请求: 180" in output
    assert "稳定性请求: 90" in output
    assert "Kimi 最短预计" not in output


def test_missing_keys_lists_environment_names_without_values(capsys):
    secret = "must-not-appear"
    exit_code = main(
        [
            "--dry-run",
            "--dataset",
            str(PROJECT_ROOT / "data" / "base_90.jsonl"),
        ],
        environ={"GLM_API_KEY": secret},
    )

    error = capsys.readouterr().err
    assert exit_code == 2
    assert "KIMI_API_KEY" in error
    assert "DEEPSEEK_API_KEY" in error
    assert secret not in error
