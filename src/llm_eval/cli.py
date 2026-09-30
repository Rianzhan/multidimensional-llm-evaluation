"""Command-line orchestration for the V2 evaluator."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

from .config import load_model_configs
from .datasets import load_jsonl
from .paths import resolve_project_root
from .reporting import generate_outputs
from .rate_limit import ProviderRateLimiter
from .runner import evaluate_suite
from .schema import EvalCase


PROJECT_ROOT = resolve_project_root(
    source_checkout=Path(__file__).resolve().parents[2]
)
MODEL_NAMES = ("GLM", "KIMI", "DEEPSEEK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多维大模型评测与选型系统 V2")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "base_90.jsonl")
    parser.add_argument(
        "--stability-dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "stability_15.jsonl",
    )
    parser.add_argument("--stability-runs", type=int, default=3)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "latest")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-robustness",
        action="store_true",
        help="Also run the separate 30-case robustness suite.",
    )
    parser.add_argument(
        "--robustness-dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "robustness_30.jsonl",
    )
    return parser.parse_args(argv)


def _stability_cases(cases: list[EvalCase], run_index: int) -> tuple[list[EvalCase], dict[str, str]]:
    repeated = []
    original_by_repeat = {}
    for case in cases:
        value = case.to_dict()
        repeat_id = f"{case.id}::run-{run_index}"
        value["id"] = repeat_id
        value["source_id"] = f"{case.source_id}::run-{run_index}"
        value["derived_from"] = case.id
        value["metadata"] = {**value.get("metadata", {}), "stability_run": run_index}
        repeated_case = EvalCase.from_dict(value)
        repeated.append(repeated_case)
        original_by_repeat[repeat_id] = case.id
    return repeated, original_by_repeat


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    runner_fn: Callable = evaluate_suite,
) -> int:
    args = parse_args(argv)
    if args.stability_runs < 1:
        print("配置错误: --stability-runs 必须至少为 1", file=sys.stderr)
        return 2
    try:
        configs = load_model_configs(
            PROJECT_ROOT / "config" / "models.json",
            os.environ if environ is None else environ,
            set(args.models),
        )
        cases = load_jsonl(args.dataset)
        stability_cases = (
            [] if args.skip_stability else load_jsonl(args.stability_dataset)
        )
        robustness_cases = (
            load_jsonl(args.robustness_dataset) if args.run_robustness else []
        )
    except (OSError, ValueError) as exc:
        print(f"配置或数据错误: {exc}", file=sys.stderr)
        return 2

    base_requests = len(cases) * len(configs)
    stability_requests = len(stability_cases) * args.stability_runs * len(configs)
    robustness_requests = len(robustness_cases) * len(configs)
    if args.dry_run:
        print(f"模型: {', '.join(config.name for config in configs)}")
        print(f"基础题: {len(cases)}")
        print(f"基础请求: {base_requests}")
        print(f"稳定性请求: {stability_requests}")
        print(f"鲁棒性请求: {robustness_requests}")
        kimi = next((config for config in configs if config.name == "KIMI"), None)
        if kimi:
            kimi_requests = len(cases) + len(stability_cases) * args.stability_runs + len(robustness_cases)
            minutes = kimi_requests * kimi.min_request_interval_seconds / 60
            print(f"Kimi 最短预计: {minutes:.1f} 分钟（未含重试）")
        print("dry-run 完成：未发送任何 API 请求。")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_limiters = {
        config.name: ProviderRateLimiter(config.min_request_interval_seconds)
        for config in configs
    }
    checkpoint = args.checkpoint or args.output_dir / "base_checkpoint.jsonl"
    if checkpoint.exists() and not args.resume:
        print("运行错误: checkpoint 已存在；请加 --resume 或指定新的输出目录。", file=sys.stderr)
        return 2
    try:
        results = runner_fn(
            cases,
            configs,
            checkpoint,
            progress=print,
            limiters=shared_limiters,
        )
        stability_results = []
        for run_index in range(1, args.stability_runs + 1):
            if not stability_cases:
                break
            repeated, original_by_repeat = _stability_cases(stability_cases, run_index)
            repeated_results = runner_fn(
                repeated,
                configs,
                args.output_dir / f"stability_checkpoint_{run_index}.jsonl",
                progress=print,
                limiters=shared_limiters,
            )
            stability_results.extend(
                replace(result, case_id=original_by_repeat[result.case_id])
                for result in repeated_results
            )
        generate_outputs(args.output_dir, results, stability_results=stability_results)
        if robustness_cases:
            robustness_results = runner_fn(
                robustness_cases,
                configs,
                args.output_dir / "robustness_checkpoint.jsonl",
                progress=print,
                limiters=shared_limiters,
            )
            generate_outputs(
                args.output_dir / "robustness",
                robustness_results,
                stability_results=[],
            )
    except Exception as exc:
        print(f"评测运行失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"评测完成，产物目录: {args.output_dir}")
    return 0
