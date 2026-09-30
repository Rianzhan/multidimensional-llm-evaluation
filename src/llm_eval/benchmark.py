"""Deterministic benchmark selection and source-to-schema adapters."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .schema import EvalCase, TaskType


SEED = 20260928
MMLU_SUBJECTS = ("math", "computer science", "physics", "health", "law")
MMLU_SOURCE_URL = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"
IFEVAL_SOURCE_URL = (
    "https://github.com/google-research/google-research/tree/"
    "d36068b845da4c2b24927fee2cea1e6ef98dadda/instruction_following_eval"
)
URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)


def _seeded_rank(seed: int, value: object) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _mmlu_complexity(row: Mapping[str, Any]) -> int:
    question = str(row.get("question", ""))
    options = row.get("options") or []
    cot = str(row.get("cot_content", ""))
    return len(question) + sum(len(str(option)) for option in options) + len(cot) * 2


def _adapt_mmlu(row: Mapping[str, Any], complexity: int) -> EvalCase:
    original_options = list(row["options"])
    answer_index = int(row["answer_index"])
    kept_indices = [
        index
        for index, option in enumerate(original_options)
        if str(option).strip().upper() != "N/A"
    ]
    if answer_index not in kept_indices:
        raise ValueError(f"MMLU-Pro answer points to removed N/A: {row['question_id']}")
    choices = tuple(str(original_options[index]).strip() for index in kept_indices)
    remapped_index = kept_indices.index(answer_index)
    reference = "ABCDEFGHIJ"[remapped_index]
    source_id = str(row["question_id"])
    return EvalCase(
        id=f"mmlu-pro-{str(row['category']).replace(' ', '-')}-{source_id}",
        task_type=TaskType.MULTIPLE_CHOICE,
        subject=str(row["category"]),
        difficulty="extreme" if complexity >= 2500 else "hard",
        source_dataset="MMLU-Pro",
        source_id=source_id,
        source_split=str(row.get("_split", "test")),
        source_url=MMLU_SOURCE_URL,
        prompt=str(row["question"]).strip(),
        choices=choices,
        reference_answer=reference,
        grader={"type": "strict_choice"},
        metadata={
            "upstream_subject": row.get("src"),
            "selection_complexity": complexity,
            "original_answer_index": answer_index,
        },
    )


def select_mmlu_pro(
    rows: Iterable[Mapping[str, Any]], seed: int = SEED
) -> list[EvalCase]:
    """Select 10 high-complexity cases per target subject without model feedback."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        category = str(row.get("category", ""))
        if row.get("_split", "test") == "test" and category in MMLU_SUBJECTS:
            grouped[category].append(row)

    selected: list[EvalCase] = []
    for subject in MMLU_SUBJECTS:
        candidates = grouped.get(subject, [])
        if len(candidates) < 10:
            raise ValueError(f"MMLU-Pro subject {subject} has fewer than 10 candidates")
        ranked = sorted(
            candidates,
            key=lambda row: (
                -_mmlu_complexity(row),
                _seeded_rank(seed, row.get("question_id")),
                str(row.get("question_id")),
            ),
        )
        variable_choice = sorted(
            (
                row
                for row in candidates
                if len(
                    [
                        option
                        for option in row.get("options", [])
                        if str(option).strip().upper() != "N/A"
                    ]
                )
                < 10
            ),
            key=lambda row: (
                len(
                    [
                        option
                        for option in row.get("options", [])
                        if str(option).strip().upper() != "N/A"
                    ]
                ),
                -_mmlu_complexity(row),
                _seeded_rank(seed, row.get("question_id")),
            ),
        )
        chosen = variable_choice[:1]
        chosen_ids = {str(row["question_id"]) for row in chosen}
        chosen.extend(
            row
            for row in ranked
            if str(row["question_id"]) not in chosen_ids
        )
        chosen = chosen[:10]
        selected.extend(_adapt_mmlu(row, _mmlu_complexity(row)) for row in chosen)
    return selected


def _instruction_prefixes(row: Mapping[str, Any]) -> set[str]:
    values = row.get("instruction_id_list") or []
    return {str(value).split(":", 1)[0] for value in values}


def select_ifeval(
    rows: Iterable[Mapping[str, Any]], seed: int = SEED
) -> list[EvalCase]:
    candidates = [
        row
        for row in rows
        if not URL_PATTERN.search(str(row.get("prompt", "")))
        and row.get("instruction_id_list")
    ]
    candidates.sort(
        key=lambda row: (_seeded_rank(seed, row.get("key")), str(row.get("key")))
    )
    if len(candidates) < 20:
        raise ValueError("IFEval has fewer than 20 URL-free candidates")

    chosen: list[Mapping[str, Any]] = []
    covered: set[str] = set()
    remaining = list(candidates)
    while len(chosen) < 20:
        best = min(
            remaining,
            key=lambda row: (
                -len(_instruction_prefixes(row) - covered),
                -len(row.get("instruction_id_list") or []),
                _seeded_rank(seed, row.get("key")),
                str(row.get("key")),
            ),
        )
        chosen.append(best)
        covered.update(_instruction_prefixes(best))
        remaining.remove(best)

    output = []
    for row in chosen:
        source_id = str(row["key"])
        instruction_ids = list(row["instruction_id_list"])
        kwargs = list(row.get("kwargs") or [])
        output.append(
            EvalCase(
                id=f"ifeval-{source_id}",
                task_type=TaskType.INSTRUCTION_FOLLOWING,
                subject="instruction_following",
                difficulty=("extreme" if len(instruction_ids) >= 3 else "hard"),
                source_dataset="IFEval",
                source_id=source_id,
                source_split="test",
                source_url=IFEVAL_SOURCE_URL,
                prompt=str(row["prompt"]),
                choices=(),
                reference_answer=None,
                grader={
                    "type": "official_ifeval",
                    "instruction_id_list": instruction_ids,
                    "kwargs": kwargs,
                },
                metadata={"instruction_prefixes": sorted(_instruction_prefixes(row))},
            )
        )
    return output


def select_stability(mmlu_cases: Iterable[EvalCase]) -> list[EvalCase]:
    grouped: dict[str, list[EvalCase]] = defaultdict(list)
    for case in mmlu_cases:
        if case.subject in MMLU_SUBJECTS:
            grouped[case.subject].append(case)
    stability = []
    for subject in MMLU_SUBJECTS:
        cases = sorted(grouped[subject], key=lambda case: case.id)
        if len(cases) < 3:
            raise ValueError(f"stability subset needs 3 cases for {subject}")
        stability.extend(cases[:3])
    return stability
