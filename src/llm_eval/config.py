"""Configuration loading with environment-only API credentials."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class TokenBudget:
    """Initial and truncation-recovery output limits for a task type."""

    initial: int
    recovery: int

    def __post_init__(self) -> None:
        if self.initial <= 0 or self.recovery < self.initial:
            raise ValueError("token budgets require 0 < initial <= recovery")


def _default_budgets() -> Mapping[str, TokenBudget]:
    return MappingProxyType(
        {
            "multiple_choice": TokenBudget(initial=64, recovery=256),
            "ifeval": TokenBudget(initial=2048, recovery=4096),
            "product_rubric": TokenBudget(initial=1024, recovery=2048),
        }
    )


@dataclass(frozen=True)
class ModelConfig:
    """Runtime settings for one OpenAI-compatible provider."""

    name: str
    api_key: str
    model: str
    base_url: str
    request_timeout_seconds: float = 60.0
    min_request_interval_seconds: float = 1.0
    max_attempts: int = 3
    output_budgets_by_task_type: Mapping[str, TokenBudget] = field(
        default_factory=_default_budgets
    )
    api_key_env: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.model.strip() or not self.base_url.strip():
            raise ValueError("model name, model id, and base URL are required")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if self.min_request_interval_seconds < 0:
            raise ValueError("minimum request interval cannot be negative")
        if self.max_attempts < 1:
            raise ValueError("maximum attempts must be at least one")


def _parse_budgets(raw: object, *, model_name: str) -> Mapping[str, TokenBudget]:
    if not isinstance(raw, dict):
        raise ValueError(f"{model_name}: output_budgets_by_task_type must be an object")

    budgets: dict[str, TokenBudget] = {}
    for task_type in ("multiple_choice", "ifeval", "product_rubric"):
        item = raw.get(task_type)
        if not isinstance(item, dict):
            raise ValueError(f"{model_name}: missing token budget for {task_type}")
        try:
            budgets[task_type] = TokenBudget(
                initial=int(item["initial"]),
                recovery=int(item["recovery"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{model_name}: invalid token budget for {task_type}"
            ) from exc
    return MappingProxyType(budgets)


def load_model_configs(
    path: Path,
    environ: Mapping[str, str],
    model_names: set[str] | None = None,
) -> list[ModelConfig]:
    """Load non-secret settings from JSON and credentials from ``environ``."""

    try:
        raw_models = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取模型配置文件: {path}") from exc

    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("模型配置必须是非空数组")

    missing: list[str] = []
    configs: list[ModelConfig] = []
    for item in raw_models:
        if not isinstance(item, dict):
            raise ValueError("每个模型配置必须是对象")
        name = str(item.get("name", "")).strip()
        if model_names is not None and name not in model_names:
            continue
        key_env = str(item.get("api_key_env", "")).strip()
        api_key = environ.get(key_env, "").strip() if key_env else ""
        if not api_key:
            missing.append(f"{name or '未命名模型'} ({key_env or '未配置环境变量名'})")
            continue

        configs.append(
            ModelConfig(
                name=name,
                api_key=api_key,
                api_key_env=key_env,
                model=str(item.get("model", "")).strip(),
                base_url=str(item.get("base_url", "")).strip(),
                request_timeout_seconds=float(
                    item.get("request_timeout_seconds", 60.0)
                ),
                min_request_interval_seconds=float(
                    item.get("min_request_interval_seconds", 1.0)
                ),
                max_attempts=int(item.get("max_attempts", 3)),
                output_budgets_by_task_type=_parse_budgets(
                    item.get("output_budgets_by_task_type"), model_name=name
                ),
            )
        )

    if missing:
        raise ValueError("缺少 API Key 环境变量: " + "; ".join(missing))
    if not configs:
        raise ValueError("没有选中可运行的模型配置")
    return configs
