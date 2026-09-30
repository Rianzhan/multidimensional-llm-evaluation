"""Prompt construction for supported evaluation task types."""

from __future__ import annotations

from .schema import EvalCase, TaskType


def build_prompt(case: EvalCase) -> list[dict[str, str]]:
    if case.task_type is not TaskType.MULTIPLE_CHOICE:
        return [{"role": "user", "content": case.prompt}]

    labels = case.allowed_answers
    option_lines = "\n".join(
        f"{label}. {choice}" for label, choice in zip(labels, case.choices)
    )
    content = (
        f"请从以下选项中选择一个答案，只输出单个大写字母（A-{labels[-1]}）：\n"
        f"题目：{case.prompt}\n"
        f"选项：\n{option_lines}"
    )
    return [{"role": "user", "content": content}]
