"""Thin adapter around the pinned official Google Research IFEval code."""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path
from typing import Any

from .schema import EvalCase, GradeResult, RunStatus, TaskType
from .paths import resolve_project_root


IFEVAL_REVISION = "d36068b845da4c2b24927fee2cea1e6ef98dadda"


def load_official_evaluator() -> Any:
    """Load the vendored source unchanged under its upstream package name."""

    package_name = "instruction_following_eval"
    project_root = resolve_project_root(
        source_checkout=Path(__file__).resolve().parents[2]
    )
    source_root = project_root / "third_party" / "ifeval"
    project_nltk_data = project_root / "data" / "nltk_data"
    os.environ.setdefault("NLTK_DATA", str(project_nltk_data))
    if not source_root.exists():
        raise RuntimeError(
            "IFEval 官方代码缺失；请先运行 python scripts/fetch_sources.py"
        )
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source_root)]
        package.__package__ = package_name
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        sys.modules[package_name] = package
    try:
        return importlib.import_module(f"{package_name}.evaluation_lib")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "IFEval 依赖缺失；请运行 pip install -r requirements.txt"
        ) from exc


def grade_ifeval(case: EvalCase, response: str) -> GradeResult:
    if case.task_type is not TaskType.INSTRUCTION_FOLLOWING:
        raise ValueError("grade_ifeval requires an instruction-following case")
    official = load_official_evaluator()
    grader = case.grader
    inp = official.InputExample(
        key=int(case.source_id),
        instruction_id_list=list(grader["instruction_id_list"]),
        prompt=case.prompt,
        kwargs=list(grader["kwargs"]),
    )
    mapping = {case.prompt: response}
    try:
        strict = official.test_instruction_following_strict(inp, mapping)
        loose = official.test_instruction_following_loose(inp, mapping)
    except LookupError as exc:
        raise RuntimeError(
            "IFEval 缺少 NLTK 资源；请运行 python -m nltk.downloader punkt punkt_tab"
        ) from exc
    followed = bool(strict.follow_all_instructions)
    return GradeResult(
        status=RunStatus.CORRECT if followed else RunStatus.WRONG,
        is_correct=followed,
        normalized_answer=None,
        details={
            "evaluator": "Google Research IFEval",
            "evaluator_revision": IFEVAL_REVISION,
            "strict": {
                "prompt_followed": followed,
                "instruction_followed": list(strict.follow_instruction_list),
            },
            "loose": {
                "prompt_followed": bool(loose.follow_all_instructions),
                "instruction_followed": list(loose.follow_instruction_list),
            },
        },
    )
