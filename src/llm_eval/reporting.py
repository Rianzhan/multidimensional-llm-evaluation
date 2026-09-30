"""JSON artifacts, charts, and a Chinese decision report."""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path
from typing import Iterable

from .metrics import mcnemar_exact, stability_summary, summarize_runs
from .providers import CaseRunResult


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _font_name() -> str:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC"):
        if candidate in available:
            return candidate
    warnings.warn("未找到 SimHei/微软雅黑，图表回退到 DejaVu Sans。", stacklevel=2)
    return "DejaVu Sans"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [_font_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def pairwise_significance(results: Iterable[CaseRunResult]) -> dict[str, dict]:
    grouped: dict[str, list[CaseRunResult]] = {}
    for result in results:
        grouped.setdefault(result.model_name, []).append(result)
    comparisons = {}
    for first, second in itertools.combinations(sorted(grouped), 2):
        comparisons[f"{first} vs {second}"] = mcnemar_exact(
            grouped[first], grouped[second]
        )
    return comparisons


def plot_accuracy_ci(path: Path, summary: dict) -> None:
    plt = _pyplot()
    tasks = list(summary["by_task_type"])
    models = sorted(summary["by_model"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    offsets = {
        model: (index - (len(models) - 1) / 2) * 0.18
        for index, model in enumerate(models)
    }
    for model in models:
        xs, ys, lower, upper = [], [], [], []
        for index, task in enumerate(tasks):
            metric = summary["by_task_type"][task].get(model)
            if not metric or metric["content_accuracy"] is None:
                continue
            value = metric["content_accuracy"]
            lo, hi = metric["content_accuracy_ci95"]
            xs.append(index + offsets[model])
            ys.append(value)
            lower.append(value - lo)
            upper.append(hi - value)
        if xs:
            ax.errorbar(xs, ys, yerr=[lower, upper], marker="o", capsize=4, label=model)
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("内容准确率（95% CI）")
    ax.set_title("分题型内容准确率与 Wilson 置信区间")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_error_heatmap(path: Path, summary: dict) -> None:
    import numpy as np

    plt = _pyplot()
    models = sorted(summary["by_model"])
    metrics = ("format_error_rate", "api_error_rate", "truncation_rate", "grader_error_rate")
    labels = ("格式", "API", "截断", "评分器")
    matrix = np.array(
        [[summary["by_model"][model].get(metric) or 0 for metric in metrics] for model in models]
    )
    fig, ax = plt.subplots(figsize=(8, max(3, len(models) * 0.8)))
    image = ax.imshow(matrix, vmin=0, vmax=max(0.01, float(matrix.max())), cmap="Reds")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(models)), models)
    for row in range(len(models)):
        for column in range(len(labels)):
            ax.text(column, row, f"{matrix[row, column]:.1%}", ha="center", va="center")
    ax.set_title("最终状态错误率热力图")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quality_latency(path: Path, summary: dict) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, metric in sorted(summary["by_model"].items()):
        quality = metric["end_to_end_success_rate"] or 0
        latency = metric["latency_ms"]["p50"] or 0
        ax.scatter(latency, quality, s=90)
        ax.annotate(model, (latency, quality), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("p50 延迟（ms）")
    ax.set_ylabel("端到端成功率")
    ax.set_ylim(0, 1.05)
    ax.set_title("质量—延迟视图（价格缺失时不估算成本）")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_report(summary: dict, significance: dict, stability: dict) -> str:
    lines = [
        "# 多维大模型评测与选型报告 V2",
        "",
        "> 结论按题型分别呈现；不可比题型不会合成为单一总分。",
        "",
        "## 运行健康与接口质量",
        "",
        "| 模型 | 端到端成功率 | 格式错误率 | API 错误率 | 截断率 | p50 延迟 | Prompt/Completion Token | 成本 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, metric in sorted(summary["by_model"].items()):
        tokens = metric["tokens"]
        lines.append(
            f"| {model} | {_percent(metric['end_to_end_success_rate'])} | "
            f"{_percent(metric['format_error_rate'])} | {_percent(metric['api_error_rate'])} | "
            f"{_percent(metric['truncation_rate'])} | {metric['latency_ms']['p50'] or 0:.0f} ms | "
            f"{tokens['prompt']}/{tokens['completion']} | {metric['cost_status']} |"
        )

    lines.extend(["", "## 分题型内容准确率与 95% 置信区间", ""])
    for task, models in sorted(summary["by_task_type"].items()):
        lines.extend(
            [
                f"### {task}",
                "",
                "| 模型 | 内容准确率 | 95% 置信区间 | 可评分样本 |",
                "|---|---:|---:|---:|",
            ]
        )
        for model, metric in sorted(models.items()):
            lo, hi = metric["content_accuracy_ci95"]
            interval = "N/A" if lo is None else f"[{lo:.2%}, {hi:.2%}]"
            lines.append(
                f"| {model} | {_percent(metric['content_accuracy'])} | {interval} | "
                f"{metric['content_evaluable']} |"
            )
        lines.append("")

    lines.extend(["## 配对显著性（exact McNemar）", ""])
    if significance:
        lines.extend(
            [
                "| 对比 | 配对题数 | 分歧对 | p 值 |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, item in significance.items():
            lines.append(
                f"| {name} | {item['paired_cases']} | {item['discordant_pairs']} | "
                f"{item['p_value']:.4f} |"
            )
    else:
        lines.append("暂无足够的两模型同题配对结果。")

    lines.extend(["", "## 稳定性", ""])
    if stability:
        for model, item in sorted(stability.items()):
            lines.append(
                f"- {model}: {item['complete_case_count']}/{item['case_count']} 道完成重复运行，"
                f"平均答案一致率 {_percent(item['mean_answer_agreement'])}。"
            )
    else:
        lines.append("尚未运行稳定性子集；其重复结果不会重复计入基础准确率。")

    lines.extend(
        [
            "",
            "## 延迟与 Token",
            "",
            "延迟同时报告 p50/p95，Token 来自供应商 usage；未配置公开价格时成本明确记为 unavailable，不猜测价格。",
            "",
            "## 场景选型建议",
            "",
        ]
    )
    for task, models in sorted(summary["by_task_type"].items()):
        candidates = [
            (model, metric["content_accuracy"], metric["end_to_end_success_rate"])
            for model, metric in models.items()
            if metric["content_accuracy"] is not None
        ]
        if candidates:
            winner = max(candidates, key=lambda item: (item[1], item[2] or 0))
            lines.append(
                f"- `{task}`：当前样本上 {winner[0]} 的内容准确率最高（{winner[1]:.2%}）；"
                "需结合置信区间、显著性与接口成功率复核。"
            )
    lines.extend(
        [
            "",
            "## 局限性",
            "",
            "- MMLU-Pro 每学科仅 10 道，必须结合 Wilson 区间理解小样本波动。",
            "- 公开基准可能存在训练污染；结果不等同于真实业务泛化能力。",
            "- IFEval 检查可验证的表面约束，不代表事实正确性。",
            "- 产品 Rubric 是可执行草案，关键应用仍需人工复核；报告保留该标记。",
            "- API、格式和截断故障与模型知识错误分开统计，避免把接口故障伪装成错题。",
            "",
        ]
    )
    return "\n".join(lines)


def generate_outputs(
    output_dir: Path,
    results: list[CaseRunResult],
    *,
    stability_results: list[CaseRunResult],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_runs(results)
    significance = pairwise_significance(results)
    stability = stability_summary(stability_results) if stability_results else {}
    enriched_summary = {
        **summary,
        "pairwise_mcnemar": significance,
        "stability": stability,
    }
    results_payload = {
        "schema_version": 2,
        "results": [result.to_dict() for result in results],
        "stability_results": [result.to_dict() for result in stability_results],
    }
    (output_dir / "results_v2.json").write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary_v2.json").write_text(
        json.dumps(enriched_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = render_report(summary, significance, stability)
    (output_dir / "evaluation_report_v2.md").write_text(report, encoding="utf-8")
    plot_accuracy_ci(output_dir / "accuracy_ci.png", summary)
    plot_error_heatmap(output_dir / "error_heatmap.png", summary)
    plot_quality_latency(output_dir / "quality_latency.png", summary)
    return {"summary": enriched_summary, "report": report}
