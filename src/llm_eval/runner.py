"""Concurrent provider queues with append-only checkpoint recovery."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Mapping

from openai import OpenAI

from .config import ModelConfig
from .providers import CaseRunResult, run_case_with_recovery
from .rate_limit import ProviderRateLimiter, retry_delay
from .schema import EvalCase, RunStatus


def _checkpoint_key(result: CaseRunResult) -> tuple[str, str]:
    return result.model_name, result.case_id


def load_checkpoint(path: Path) -> dict[tuple[str, str], CaseRunResult]:
    latest: dict[tuple[str, str], CaseRunResult] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
            result = CaseRunResult.from_dict(value)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if result.attempts:
            latest[_checkpoint_key(result)] = result
    return latest


def _combine_runs(runs: list[CaseRunResult]) -> CaseRunResult:
    first = runs[0]
    last = runs[-1]
    attempts = []
    for run in runs:
        for attempt in run.attempts:
            attempts.append(replace(attempt, attempt_number=len(attempts) + 1))
    return CaseRunResult(
        model_name=first.model_name,
        model_id=first.model_id,
        case_id=first.case_id,
        task_type=first.task_type,
        subject=first.subject,
        attempts=tuple(attempts),
        first_pass_status=first.first_pass_status,
        final_status=last.final_status,
        final_answer=last.final_answer,
        is_correct=last.is_correct,
        recovered=(
            first.first_pass_status is RunStatus.TRUNCATED
            and last.final_status is not RunStatus.TRUNCATED
        ),
    )


def evaluate_suite(
    cases: Iterable[EvalCase],
    configs: Iterable[ModelConfig],
    checkpoint_path: Path,
    *,
    run_case_fn: Callable = run_case_with_recovery,
    client_factory: Callable = OpenAI,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    wall_clock: Callable[[], float] = time.time,
    progress: Callable[[str], None] | None = None,
    limiters: Mapping[str, ProviderRateLimiter] | None = None,
) -> list[CaseRunResult]:
    """Evaluate each provider sequentially while providers run in parallel."""

    case_list = list(cases)
    config_list = list(configs)
    existing = load_checkpoint(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        current = checkpoint_path.read_bytes()
        if current and not current.endswith((b"\n", b"\r")):
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")

    write_lock = threading.Lock()
    result_map = dict(existing)

    def append_checkpoint(result: CaseRunResult) -> None:
        payload = json.dumps(result.to_dict(), ensure_ascii=False)
        with write_lock:
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
            result_map[_checkpoint_key(result)] = result

    def provider_worker(config: ModelConfig) -> None:
        limiter = (
            limiters.get(config.name)
            if limiters is not None
            else None
        ) or ProviderRateLimiter(
            config.min_request_interval_seconds, clock=clock, sleep_fn=sleep_fn
        )
        for index, case in enumerate(case_list, start=1):
            key = (config.name, case.id)
            if key in existing:
                continue

            runs: list[CaseRunResult] = []
            pending_retry_wait = 0.0
            for request_attempt in range(1, config.max_attempts + 1):
                initial_wait = pending_retry_wait + limiter.wait()
                pending_retry_wait = 0.0
                result = run_case_fn(
                    config,
                    case,
                    client_factory=client_factory,
                    clock=clock,
                    initial_wait_seconds=initial_wait,
                    recovery_wait_fn=limiter.wait,
                )
                runs.append(result)
                if result.final_status is not RunStatus.API_ERROR:
                    break
                last_attempt = result.attempts[-1]
                if last_attempt.error_type == "authentication":
                    break
                if request_attempt < config.max_attempts:
                    pending_retry_wait = retry_delay(
                        last_attempt, request_attempt, wall_clock()
                    )
                    sleep_fn(pending_retry_wait)

            combined = _combine_runs(runs)
            append_checkpoint(combined)
            if progress:
                progress(
                    f"{config.name}: {index}/{len(case_list)} "
                    f"{case.id} -> {combined.final_status.value}"
                )

    if config_list:
        with ThreadPoolExecutor(max_workers=len(config_list)) as executor:
            futures = [executor.submit(provider_worker, config) for config in config_list]
            for future in futures:
                future.result()

    case_order = {case.id: index for index, case in enumerate(case_list)}
    model_order = {config.name: index for index, config in enumerate(config_list)}
    in_scope = [
        result
        for key, result in result_map.items()
        if key[0] in model_order and key[1] in case_order
    ]
    return sorted(
        in_scope,
        key=lambda item: (case_order[item.case_id], model_order[item.model_name]),
    )
