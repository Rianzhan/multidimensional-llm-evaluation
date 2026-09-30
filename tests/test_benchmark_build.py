from llm_eval.benchmark import select_ifeval, select_mmlu_pro, select_stability


TARGETS = ("math", "computer science", "physics", "health", "law")


def mmlu_rows():
    rows = []
    question_id = 1
    for category in TARGETS:
        for index in range(14):
            options = [f"option {letter}" for letter in "ABCDEFGHIJ"]
            answer_index = index % 10
            if index == 0:
                options = ["first", "N/A", "answer", "last"]
                answer_index = 2
            rows.append(
                {
                    "_split": "test",
                    "question_id": question_id,
                    "category": category,
                    "question": f"{category} difficult question {index} " + "detail " * index,
                    "options": options,
                    "answer": "ABCDEFGHIJ"[answer_index],
                    "answer_index": answer_index,
                    "cot_content": "step " * (index + 1),
                    "src": "fixture",
                }
            )
            question_id += 1
    return rows


def ifeval_rows():
    prefixes = [
        "length_constraints",
        "detectable_format",
        "keywords",
        "punctuation",
        "language",
        "startend",
        "change_case",
        "combination",
    ]
    rows = []
    for index in range(30):
        prefix = prefixes[index % len(prefixes)]
        rows.append(
            {
                "key": index,
                "prompt": f"Instruction following prompt {index}",
                "instruction_id_list": [f"{prefix}:rule_{index}"],
                "kwargs": [{"value": index}],
            }
        )
    rows.append(
        {
            "key": 999,
            "prompt": "Read https://example.test and summarize it",
            "instruction_id_list": ["web:requires_url"],
            "kwargs": [{}],
        }
    )
    return rows


def test_mmlu_selection_has_ten_per_target_and_unique_sources():
    selected = select_mmlu_pro(mmlu_rows(), seed=20260928)
    counts = {subject: 0 for subject in TARGETS}
    for case in selected:
        counts[case.subject] += 1

    assert len(selected) == 50
    assert counts == {subject: 10 for subject in TARGETS}
    assert len({case.source_id for case in selected}) == 50


def test_mmlu_selection_is_deterministic_and_does_not_use_model_results():
    forward = select_mmlu_pro(mmlu_rows(), seed=20260928)
    reverse = select_mmlu_pro(list(reversed(mmlu_rows())), seed=20260928)

    assert [case.source_id for case in forward] == [case.source_id for case in reverse]


def test_mmlu_answer_is_remapped_after_na_cleanup():
    selected = select_mmlu_pro(mmlu_rows(), seed=20260928)
    cleaned = [case for case in selected if "N/A" not in case.choices and len(case.choices) == 3]

    assert cleaned
    assert all(case.reference_answer == "B" for case in cleaned)
    assert all(case.reference_answer in case.allowed_answers for case in selected)


def test_ifeval_selection_has_twenty_unique_url_free_cases():
    selected = select_ifeval(ifeval_rows(), seed=20260928)

    assert len(selected) == 20
    assert len({case.source_id for case in selected}) == 20
    assert all("http" not in case.prompt for case in selected)
    assert all(case.grader["instruction_id_list"] for case in selected)


def test_ifeval_selection_is_reproducible():
    first = select_ifeval(ifeval_rows(), seed=20260928)
    second = select_ifeval(list(reversed(ifeval_rows())), seed=20260928)

    assert [case.source_id for case in first] == [case.source_id for case in second]


def test_stability_subset_has_three_cases_per_mmlu_subject():
    selected = select_mmlu_pro(mmlu_rows(), seed=20260928)
    stability = select_stability(selected)

    assert len(stability) == 15
    assert {subject: sum(c.subject == subject for c in stability) for subject in TARGETS} == {
        subject: 3 for subject in TARGETS
    }
