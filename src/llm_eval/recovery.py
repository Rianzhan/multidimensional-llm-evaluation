"""Post-hoc recovery experiments for originally truncated model-case pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from collections import Counter, defaultdict
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from openai import OpenAI

from .config import ModelConfig, TokenBudget
from .datasets import load_jsonl
from .providers import CaseRunResult, call_once
from .runner import evaluate_suite, load_checkpoint
from .schema import EvalCase, RunStatus, TaskType


@dataclass(frozen=True)
class RecoveryTarget:
    suite: str
    result: CaseRunResult


def _checkpoint_specs(source_run: Path) -> list[tuple[str, Path]]:
    checkpoints: list[tuple[str, Path]] = [
        ("base", source_run / "base_checkpoint.jsonl"),
    ]
    checkpoints.extend(
        (f"stability-{index}", source_run / f"stability_checkpoint_{index}.jsonl")
        for index in range(1, 4)
    )
    checkpoints.append(("robustness", source_run / "robustness_checkpoint.jsonl"))
    return checkpoints


def collect_source_results(source_run: Path) -> list[RecoveryTarget]:
    """Load every latest formal result and preserve its originating suite."""

    results: list[RecoveryTarget] = []
    for suite, path in _checkpoint_specs(source_run):
        if not path.exists():
            continue
        for result in load_checkpoint(path).values():
            results.append(RecoveryTarget(suite=suite, result=result))
    return results


def collect_recovery_targets(source_run: Path) -> list[RecoveryTarget]:
    """Return final truncations from every checkpoint in stable suite order."""

    return [
        item
        for item in collect_source_results(source_run)
        if item.result.final_status is RunStatus.TRUNCATED
    ]


def load_recovery_run(recovery_run: Path) -> tuple[Path, list[CaseRunResult]]:
    """Load an earlier recovery artifact for a bounded follow-up retry."""

    path = recovery_run / "recovery_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_run = payload.get("source_run")
    raw_results = payload.get("results")
    if not isinstance(source_run, str) or not source_run.strip():
        raise ValueError(f"recovery artifact has no source_run: {path}")
    if not isinstance(raw_results, list):
        raise ValueError(f"recovery artifact has no results list: {path}")
    return (
        Path(source_run).resolve(),
        [CaseRunResult.from_dict(item) for item in raw_results],
    )


def load_recovery_budget_policy(recovery_run: Path) -> dict[str, int]:
    """Read the declared budget policy from an earlier recovery artifact."""

    path = recovery_run / "recovery_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = payload.get("budget_policy")
    if not isinstance(policy, dict):
        raise ValueError(f"recovery artifact has no budget_policy: {path}")
    required = ("multiple_choice", "instruction_following")
    if any(not isinstance(policy.get(key), int) for key in required):
        raise ValueError(f"recovery artifact has an invalid budget_policy: {path}")
    return {str(key): int(value) for key, value in policy.items()}


def collect_remaining_recovery_targets(
    recovery_results: Iterable[CaseRunResult],
) -> list[RecoveryTarget]:
    """Select only records still truncated after an earlier recovery run."""

    return [
        RecoveryTarget(suite="final-retry", result=result)
        for result in recovery_results
        if result.final_status is RunStatus.TRUNCATED
    ]


def merge_recovery_results(
    prior_results: Iterable[CaseRunResult],
    retry_results: Iterable[CaseRunResult],
) -> list[CaseRunResult]:
    """Replace retried model-case pairs while preserving the prior result order."""

    prior_list = list(prior_results)
    prior_keys = {(result.model_name, result.case_id) for result in prior_list}
    replacements = {
        (result.model_name, result.case_id): result for result in retry_results
    }
    unknown = sorted(set(replacements) - prior_keys)
    if unknown:
        model_name, case_id = unknown[0]
        raise ValueError(f"retry result was not present in prior recovery: {model_name} {case_id}")
    return [
        replacements.get((result.model_name, result.case_id), result)
        for result in prior_list
    ]


def _repeat_stability_case(case: EvalCase, run_index: int) -> EvalCase:
    value = case.to_dict()
    value["id"] = f"{case.id}::run-{run_index}"
    value["source_id"] = f"{case.source_id}::run-{run_index}"
    value["derived_from"] = case.id
    value["metadata"] = {**value.get("metadata", {}), "stability_run": run_index}
    return EvalCase.from_dict(value)


def build_case_catalog(data_dir: Path, *, stability_runs: int = 3) -> dict[str, EvalCase]:
    """Load the formal datasets and reconstruct stability case identifiers."""

    cases = [
        *load_jsonl(data_dir / "base_90.jsonl"),
        *load_jsonl(data_dir / "robustness_30.jsonl"),
    ]
    stability = load_jsonl(data_dir / "stability_15.jsonl")
    for run_index in range(1, stability_runs + 1):
        cases.extend(_repeat_stability_case(case, run_index) for case in stability)

    catalog: dict[str, EvalCase] = {}
    for case in cases:
        if case.id in catalog:
            raise ValueError(f"duplicate recovery case id: {case.id}")
        catalog[case.id] = case
    return catalog


def with_extended_budgets(
    config: ModelConfig,
    *,
    multiple_choice_tokens: int = 4096,
    ifeval_tokens: int = 8192,
) -> ModelConfig:
    """Use the declared post-hoc budgets without changing provider settings."""

    budgets = dict(config.output_budgets_by_task_type)
    budgets["multiple_choice"] = TokenBudget(
        initial=multiple_choice_tokens,
        recovery=multiple_choice_tokens,
    )
    budgets["ifeval"] = TokenBudget(initial=ifeval_tokens, recovery=ifeval_tokens)
    return replace(
        config,
        output_budgets_by_task_type=MappingProxyType(budgets),
    )


def run_extended_case(
    config: ModelConfig,
    case: EvalCase,
    client_factory: Callable = OpenAI,
    clock: Callable[[], float] | None = None,
    *,
    initial_wait_seconds: float = 0.0,
    recovery_wait_fn: Callable[[], float] | None = None,
) -> CaseRunResult:
    """Run exactly one extended-budget response for a recovery target."""

    del recovery_wait_fn
    budget_key = {
        TaskType.MULTIPLE_CHOICE: "multiple_choice",
        TaskType.INSTRUCTION_FOLLOWING: "ifeval",
        TaskType.RUBRIC: "product_rubric",
    }[case.task_type]
    call_kwargs = {
        "max_tokens": config.output_budgets_by_task_type[budget_key].initial,
        "attempt_number": 1,
        "wait_seconds": initial_wait_seconds,
    }
    if clock is not None:
        call_kwargs["clock"] = clock
    attempt = call_once(
        config,
        case,
        client_factory,
        **call_kwargs,
    )
    return CaseRunResult(
        model_name=config.name,
        model_id=config.model,
        case_id=case.id,
        task_type=case.task_type,
        subject=case.subject,
        attempts=(attempt,),
        first_pass_status=attempt.status,
        final_status=attempt.status,
        final_answer=attempt.normalized_answer,
        is_correct=attempt.is_correct,
        recovered=False,
    )


def run_recovery_experiment(
    targets: Iterable[RecoveryTarget],
    catalog: Mapping[str, EvalCase],
    configs: Iterable[ModelConfig],
    output_dir: Path,
    *,
    runner_fn: Callable = evaluate_suite,
    progress: Callable[[str], None] | None = None,
    multiple_choice_tokens: int = 4096,
    ifeval_tokens: int = 8192,
) -> list[CaseRunResult]:
    """Evaluate only the sparse model-case pairs selected for recovery."""

    target_list = list(targets)
    config_list = list(configs)
    configured_names = {config.name for config in config_list}
    target_names = {target.result.model_name for target in target_list}
    missing_models = sorted(target_names - configured_names)
    if missing_models:
        raise ValueError("missing recovery model configs: " + ", ".join(missing_models))

    missing_cases = sorted(
        {target.result.case_id for target in target_list} - set(catalog)
    )
    if missing_cases:
        raise ValueError("missing recovery cases: " + ", ".join(missing_cases))

    output_dir.mkdir(parents=True, exist_ok=True)
    recovered: list[CaseRunResult] = []
    for config in config_list:
        model_targets = [
            target for target in target_list if target.result.model_name == config.name
        ]
        if not model_targets:
            continue
        cases = [catalog[target.result.case_id] for target in model_targets]
        checkpoint = output_dir / f"recovery_{config.name.lower()}_checkpoint.jsonl"
        recovered.extend(
            runner_fn(
                cases,
                [
                    with_extended_budgets(
                        config,
                        multiple_choice_tokens=multiple_choice_tokens,
                        ifeval_tokens=ifeval_tokens,
                    )
                ],
                checkpoint,
                run_case_fn=run_extended_case,
                progress=progress,
            )
        )
    return recovered


def build_recovery_summary(
    source_results: Iterable[RecoveryTarget],
    recovery_results: Iterable[CaseRunResult],
) -> dict:
    """Compare the immutable strict run with post-hoc recovery outcomes."""

    source_list = list(source_results)
    recovery_list = list(recovery_results)
    source_map: dict[tuple[str, str], RecoveryTarget] = {}
    for item in source_list:
        key = (item.result.model_name, item.result.case_id)
        if key in source_map:
            raise ValueError(f"duplicate source result: {key[0]} {key[1]}")
        source_map[key] = item

    target_map = {
        key: item
        for key, item in source_map.items()
        if item.result.final_status is RunStatus.TRUNCATED
    }
    recovery_map: dict[tuple[str, str], CaseRunResult] = {}
    for result in recovery_list:
        key = (result.model_name, result.case_id)
        if key not in target_map:
            raise ValueError(f"recovery result is not an original truncation: {key}")
        recovery_map[key] = result

    source_by_model: dict[str, list[CaseRunResult]] = defaultdict(list)
    for item in source_list:
        source_by_model[item.result.model_name].append(item.result)

    by_model = {}
    for model_name, original_runs in sorted(source_by_model.items()):
        model_targets = {
            key: item for key, item in target_map.items() if key[0] == model_name
        }
        completed = {
            key: result
            for key, result in recovery_map.items()
            if key[0] == model_name
        }
        effective_statuses = []
        for original in original_runs:
            key = (model_name, original.case_id)
            effective_statuses.append(
                completed[key].final_status if key in completed else original.final_status
            )

        total = len(original_runs)
        strict_correct = sum(
            result.final_status is RunStatus.CORRECT for result in original_runs
        )
        effective_correct = sum(
            status is RunStatus.CORRECT for status in effective_statuses
        )
        completed_count = len(completed)
        recovered_count = sum(
            result.final_status is not RunStatus.TRUNCATED
            for result in completed.values()
        )
        strict_accuracy = strict_correct / total if total else None
        effective_accuracy = effective_correct / total if total else None
        by_model[model_name] = {
            "total_results": total,
            "target_count": len(model_targets),
            "completed_target_count": completed_count,
            "recovered_from_truncation_count": recovered_count,
            "truncation_recovery_rate": (
                recovered_count / completed_count if completed_count else None
            ),
            "remaining_truncated_count": sum(
                result.final_status is RunStatus.TRUNCATED
                for result in completed.values()
            ),
            "recovery_correct_count": sum(
                result.final_status is RunStatus.CORRECT
                for result in completed.values()
            ),
            "recovery_format_error_count": sum(
                result.final_status is RunStatus.FORMAT_ERROR
                for result in completed.values()
            ),
            "recovery_api_error_count": sum(
                result.final_status is RunStatus.API_ERROR
                for result in completed.values()
            ),
            "strict_correct_count": strict_correct,
            "effective_correct_count": effective_correct,
            "strict_end_to_end_accuracy": strict_accuracy,
            "effective_end_to_end_accuracy": effective_accuracy,
            "accuracy_delta_percentage_points": (
                (effective_accuracy - strict_accuracy) * 100
                if strict_accuracy is not None and effective_accuracy is not None
                else None
            ),
            "strict_status_counts": dict(
                sorted(Counter(result.final_status.value for result in original_runs).items())
            ),
            "effective_status_counts": dict(
                sorted(Counter(status.value for status in effective_statuses).items())
            ),
        }

    records = []
    for key, target in target_map.items():
        recovered = recovery_map.get(key)
        records.append(
            {
                "suite": target.suite,
                "model_name": target.result.model_name,
                "case_id": target.result.case_id,
                "task_type": target.result.task_type.value,
                "original_status": target.result.final_status.value,
                "recovery_status": (
                    recovered.final_status.value if recovered is not None else "pending"
                ),
                "recovery_answer": (
                    recovered.final_answer if recovered is not None else None
                ),
                "recovery_is_correct": (
                    recovered.is_correct if recovered is not None else None
                ),
                "recovery_max_tokens": (
                    recovered.attempts[-1].max_tokens
                    if recovered is not None and recovered.attempts
                    else None
                ),
            }
        )

    return {
        "source_result_count": len(source_list),
        "target_count": len(target_map),
        "completed_target_count": len(recovery_map),
        "by_model": by_model,
        "records": records,
    }


def _percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def write_recovery_outputs(
    output_dir: Path,
    source_run: Path,
    source_results: Iterable[RecoveryTarget],
    recovery_results: Iterable[CaseRunResult],
    *,
    budget_policy: Mapping[str, int] | None = None,
) -> dict:
    """Write separate recovery artifacts without mutating the formal run."""

    source_list = list(source_results)
    recovery_list = list(recovery_results)
    summary = build_recovery_summary(source_list, recovery_list)
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_budget_policy = dict(
        budget_policy
        or {
            "multiple_choice": 4096,
            "instruction_following": 8192,
        }
    )
    results_payload = {
        "schema_version": 1,
        "experiment_type": "post_hoc_truncation_recovery",
        "source_run": str(source_run.resolve()),
        "budget_policy": effective_budget_policy,
        "results": [result.to_dict() for result in recovery_list],
    }
    (output_dir / "recovery_results.json").write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "recovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if "final_retry_multiple_choice" in effective_budget_policy:
        budget_lines = [
            "- 第一轮扩展预算：选择题 "
            f"{effective_budget_policy['multiple_choice']} tokens；指令遵循题 "
            f"{effective_budget_policy['instruction_following']} tokens",
            "- 最终重试选择题预算："
            f"{effective_budget_policy['final_retry_multiple_choice']} tokens",
        ]
    else:
        budget_lines = [
            "- 扩展预算：选择题 "
            f"{effective_budget_policy['multiple_choice']} tokens；指令遵循题 "
            f"{effective_budget_policy['instruction_following']} tokens"
        ]

    lines = [
        "# 截断后验恢复实验",
        "",
        "> 原始正式成绩保持不变；本报告仅展示扩展输出预算后的恢复结果。",
        "",
        f"- 原始结果：{summary['source_result_count']} 条",
        f"- 截断目标：{summary['target_count']} 条",
        f"- 已完成恢复：{summary['completed_target_count']} 条",
        *budget_lines,
        "",
        "## 模型恢复对比",
        "",
        "| 模型 | 截断目标 | 已补跑 | 截断恢复率 | 严格端到端正确率 | 恢复后端到端正确率 | 变化 | 格式错误 | 仍截断 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, metrics in sorted(summary["by_model"].items()):
        delta = metrics["accuracy_delta_percentage_points"]
        delta_text = "n/a" if delta is None else f"{delta:+.2f} pp"
        lines.append(
            "| "
            + " | ".join(
                (
                    model_name,
                    str(metrics["target_count"]),
                    str(metrics["completed_target_count"]),
                    _percentage(metrics["truncation_recovery_rate"]),
                    _percentage(metrics["strict_end_to_end_accuracy"]),
                    _percentage(metrics["effective_end_to_end_accuracy"]),
                    delta_text,
                    str(metrics["recovery_format_error_count"]),
                    str(metrics["remaining_truncated_count"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 明细",
            "",
            "| 数据段 | 模型 | 题目 | 原状态 | 恢复状态 | 答案 | 正确性 | 预算 |",
            "|---|---|---|---|---|---|---|---:|",
        ]
    )
    for record in summary["records"]:
        lines.append(
            "| "
            + " | ".join(
                str(value)
                for value in (
                    record["suite"],
                    record["model_name"],
                    record["case_id"],
                    record["original_status"],
                    record["recovery_status"],
                    record["recovery_answer"] or "-",
                    record["recovery_is_correct"],
                    record["recovery_max_tokens"] or "-",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 恢复结果是看到截断后进行的后验实验，不替换原始正式成绩。",
            "- 格式错误表示模型完成了输出但未遵守答案格式，不计为内容正确。",
            "- 恢复后端到端正确率仅用于分析扩展预算策略的收益。",
            "",
        ]
    )
    (output_dir / "recovery_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return {
        "summary": summary,
        "results_path": output_dir / "recovery_results.json",
        "summary_path": output_dir / "recovery_summary.json",
        "report_path": output_dir / "recovery_report.md",
    }
