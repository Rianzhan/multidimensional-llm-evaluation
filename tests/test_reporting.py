from pathlib import Path

from llm_eval.providers import AttemptRecord, CaseRunResult
from llm_eval.reporting import generate_outputs
from llm_eval.schema import RunStatus, TaskType


def make_result(case_id, model, status, task_type, subject, latency):
    correct = True if status is RunStatus.CORRECT else False if status is RunStatus.WRONG else None
    answer = "A" if status is RunStatus.CORRECT else "B" if status is RunStatus.WRONG else None
    attempt = AttemptRecord(
        model_name=model,
        model_id=f"{model}-model",
        case_id=case_id,
        attempt_number=1,
        max_tokens=64,
        status=status,
        raw_response=answer,
        normalized_answer=answer,
        is_correct=correct,
        finish_reason="length" if status is RunStatus.TRUNCATED else "stop",
        prompt_tokens=20,
        completion_tokens=2,
        total_tokens=22,
        latency_ms=latency,
        http_status=200,
    )
    return CaseRunResult(
        model_name=model,
        model_id=f"{model}-model",
        case_id=case_id,
        task_type=task_type,
        subject=subject,
        attempts=(attempt,),
        first_pass_status=status,
        final_status=status,
        final_answer=answer,
        is_correct=correct,
        recovered=False,
    )


def test_generate_outputs_writes_auditable_json_report_and_charts(tmp_path):
    results = [
        make_result("m1", "GLM", RunStatus.CORRECT, TaskType.MULTIPLE_CHOICE, "math", 100),
        make_result("i1", "GLM", RunStatus.WRONG, TaskType.INSTRUCTION_FOLLOWING, "instruction_following", 200),
        make_result("p1", "GLM", RunStatus.FORMAT_ERROR, TaskType.RUBRIC, "medical_safety", 300),
        make_result("m1", "KIMI", RunStatus.WRONG, TaskType.MULTIPLE_CHOICE, "math", 400),
        make_result("i1", "KIMI", RunStatus.CORRECT, TaskType.INSTRUCTION_FOLLOWING, "instruction_following", 500),
        make_result("p1", "KIMI", RunStatus.API_ERROR, TaskType.RUBRIC, "medical_safety", 600),
    ]

    artifacts = generate_outputs(tmp_path, results, stability_results=[])

    for name in (
        "results_v2.json",
        "summary_v2.json",
        "evaluation_report_v2.md",
        "accuracy_ci.png",
        "error_heatmap.png",
        "quality_latency.png",
    ):
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "accuracy_ci.png").read_bytes().startswith(b"\x89PNG")
    report = (tmp_path / "evaluation_report_v2.md").read_text(encoding="utf-8")
    for heading in (
        "内容准确率",
        "端到端成功率",
        "格式错误率",
        "API 错误率",
        "截断率",
        "95% 置信区间",
        "McNemar",
        "延迟与 Token",
        "局限性",
        "场景选型建议",
    ):
        assert heading in report
    assert "不可比题型不会合成为单一总分" in report
    assert "api_key" not in (tmp_path / "results_v2.json").read_text(encoding="utf-8")
    assert artifacts["summary"]["base_case_count"] == 3
