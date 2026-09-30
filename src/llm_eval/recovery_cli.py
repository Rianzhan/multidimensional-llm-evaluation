"""CLI orchestration for the post-hoc truncation recovery experiment."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping

from .config import load_model_configs
from .paths import resolve_project_root
from .recovery import (
    build_case_catalog,
    collect_remaining_recovery_targets,
    collect_recovery_targets,
    collect_source_results,
    load_recovery_budget_policy,
    load_recovery_run,
    merge_recovery_results,
    run_recovery_experiment,
    write_recovery_outputs,
)
from .runner import evaluate_suite


PROJECT_ROOT = resolve_project_root(
    source_checkout=Path(__file__).resolve().parents[2]
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc recovery for truncated formal evaluation results"
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--prior-recovery-run", type=Path)
    parser.add_argument("--multiple-choice-tokens", type=int, default=4096)
    parser.add_argument("--ifeval-tokens", type=int, default=8192)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    runner_fn: Callable = evaluate_suite,
) -> int:
    args = parse_args(argv)
    if not args.source_run.is_dir():
        print(f"Recovery error: source run not found: {args.source_run}", file=sys.stderr)
        return 2

    try:
        source_results = collect_source_results(args.source_run)
        prior_results = []
        prior_budget_policy = None
        if args.prior_recovery_run is not None:
            prior_source, prior_results = load_recovery_run(args.prior_recovery_run)
            prior_budget_policy = load_recovery_budget_policy(args.prior_recovery_run)
            if prior_source != args.source_run.resolve():
                raise ValueError(
                    "prior recovery source does not match --source-run: "
                    f"{prior_source}"
                )
            targets = collect_remaining_recovery_targets(prior_results)
        else:
            targets = collect_recovery_targets(args.source_run)
        if not source_results:
            raise ValueError("source run has no readable checkpoint results")
        if not targets:
            print("Recovery targets: 0")
            print("No truncated results require recovery.")
            return 0
        target_names = {target.result.model_name for target in targets}
        configs = load_model_configs(
            PROJECT_ROOT / "config" / "models.json",
            os.environ if environ is None else environ,
            target_names,
        )
        catalog = build_case_catalog(args.data_dir)
    except (OSError, ValueError) as exc:
        print(f"Recovery configuration error: {exc}", file=sys.stderr)
        return 2

    checkpoints = list(args.output_dir.glob("recovery_*_checkpoint.jsonl"))
    if checkpoints and not args.resume:
        print(
            "Recovery error: checkpoint exists; add --resume or use a new output directory.",
            file=sys.stderr,
        )
        return 2

    by_model = Counter(target.result.model_name for target in targets)
    if args.dry_run:
        print(f"Recovery targets: {len(targets)}")
        for model_name, count in sorted(by_model.items()):
            print(f"{model_name}: {count}")
        kimi = next((config for config in configs if config.name == "KIMI"), None)
        if kimi:
            minutes = by_model["KIMI"] * kimi.min_request_interval_seconds / 60
            print(f"Kimi minimum estimate: {minutes:.1f} minutes")
        print(f"Multiple-choice budget: {args.multiple_choice_tokens}")
        print(f"Instruction-following budget: {args.ifeval_tokens}")
        print("Dry run complete; no API requests were sent.")
        return 0

    try:
        results = run_recovery_experiment(
            targets,
            catalog,
            configs,
            args.output_dir,
            runner_fn=runner_fn,
            progress=print,
            multiple_choice_tokens=args.multiple_choice_tokens,
            ifeval_tokens=args.ifeval_tokens,
        )
        output_results = (
            merge_recovery_results(prior_results, results)
            if prior_results
            else results
        )
        output_budget_policy = {
            "multiple_choice": args.multiple_choice_tokens,
            "instruction_following": args.ifeval_tokens,
        }
        if prior_budget_policy is not None:
            output_budget_policy = {
                **prior_budget_policy,
                "final_retry_multiple_choice": args.multiple_choice_tokens,
            }
        write_recovery_outputs(
            args.output_dir,
            args.source_run,
            source_results,
            output_results,
            budget_policy=output_budget_policy,
        )
    except Exception as exc:
        print(f"Recovery run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Recovery complete: {args.output_dir}")
    return 0
