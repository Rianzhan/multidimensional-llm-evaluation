"""Portfolio-ready figures built from formal and recovery summaries."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .reporting import _pyplot


FIGURE_FILENAMES = (
    "01_model_scorecard.png",
    "02_task_accuracy_ci.png",
    "03_error_profile_heatmap.png",
    "04_recovery_outcomes.png",
)

MODEL_ORDER = ("GLM", "KIMI", "DEEPSEEK")
TASK_ORDER = ("multiple_choice", "instruction_following", "rubric")
TASK_LABELS = {
    "multiple_choice": "选择题",
    "instruction_following": "指令遵循",
    "rubric": "产品 Rubric",
}
MODEL_COLORS = {
    "GLM": "#2563EB",
    "KIMI": "#7C3AED",
    "DEEPSEEK": "#0F9D8A",
}


def prepare_portfolio_data(formal_summary: dict, recovery_summary: dict) -> dict:
    """Normalize chart inputs while preserving every metric denominator."""

    present_models = set(recovery_summary.get("by_model", {}))
    models = [model for model in MODEL_ORDER if model in present_models]
    models.extend(sorted(present_models - set(models)))

    overview = {
        model: {
            "strict": recovery_summary["by_model"][model][
                "strict_end_to_end_accuracy"
            ],
            "recovered": recovery_summary["by_model"][model][
                "effective_end_to_end_accuracy"
            ],
        }
        for model in models
    }

    tasks = {}
    for task_name in TASK_ORDER:
        task_metrics = formal_summary.get("by_task_type", {}).get(task_name)
        if not task_metrics:
            continue
        tasks[task_name] = {
            model: {
                "accuracy": task_metrics[model]["content_accuracy"],
                "ci95": task_metrics[model]["content_accuracy_ci95"],
                "n": task_metrics[model]["content_evaluable"],
            }
            for model in models
            if model in task_metrics
        }

    errors = {}
    for model in models:
        metrics = recovery_summary["by_model"][model]
        total = metrics["total_results"]
        counts = metrics.get("strict_status_counts", {})
        errors[model] = {
            "wrong": counts.get("wrong", 0) / total if total else 0.0,
            "format_error": counts.get("format_error", 0) / total if total else 0.0,
            "truncated": counts.get("truncated", 0) / total if total else 0.0,
            "api_error": counts.get("api_error", 0) / total if total else 0.0,
        }

    outcome_names = (
        "correct",
        "wrong",
        "format_error",
        "truncated",
        "api_error",
        "grader_error",
    )
    recovery_outcomes = {
        model: {status: 0 for status in outcome_names} for model in models
    }
    for record in recovery_summary.get("records", []):
        model = record["model_name"]
        status = record["recovery_status"]
        if model in recovery_outcomes:
            recovery_outcomes[model][status] = (
                recovery_outcomes[model].get(status, 0) + 1
            )

    return {
        "models": models,
        "overview": overview,
        "tasks": tasks,
        "errors": errors,
        "recovery_outcomes": recovery_outcomes,
    }


def _finish_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")


def _add_heading(ax, title: str, subtitle: str):
    """Place a left-aligned title and subtitle on separate visual rows."""

    ax.set_title(title, loc="left", weight="bold", pad=38)
    return ax.text(
        0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        color="#64748B",
        va="bottom",
    )


def _plot_scorecard(path: Path, data: dict) -> None:
    import numpy as np

    plt = _pyplot()
    models = data["models"]
    x = np.arange(len(models))
    width = 0.34
    strict = [data["overview"][model]["strict"] for model in models]
    recovered = [data["overview"][model]["recovered"] for model in models]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    bars_strict = ax.bar(
        x - width / 2,
        strict,
        width,
        label="严格预算",
        color="#CBD5E1",
        edgecolor="#64748B",
    )
    bars_recovered = ax.bar(
        x + width / 2,
        recovered,
        width,
        label="扩展预算恢复后",
        color=[MODEL_COLORS.get(model, "#475569") for model in models],
    )
    ax.bar_label(bars_strict, labels=[f"{value:.1%}" for value in strict], padding=3)
    ax.bar_label(
        bars_recovered, labels=[f"{value:.1%}" for value in recovered], padding=3
    )
    ax.set_xticks(x, models)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("端到端正确率")
    _add_heading(
        ax,
        "模型综合表现：严格成绩与截断恢复后成绩",
        "覆盖基础、稳定性与鲁棒性共 495 次模型执行",
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _finish_figure(fig, path)
    plt.close(fig)


def _plot_task_accuracy(path: Path, data: dict) -> None:
    plt = _pyplot()
    tasks = list(data["tasks"])
    models = data["models"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    spacing = 0.18
    for model_index, model in enumerate(models):
        offset = (model_index - (len(models) - 1) / 2) * spacing
        xs, ys, lower, upper = [], [], [], []
        point_labels = []
        for task_index, task_name in enumerate(tasks):
            metric = data["tasks"][task_name].get(model)
            if not metric or metric["accuracy"] is None:
                continue
            value = metric["accuracy"]
            lo, hi = metric["ci95"]
            xs.append(task_index + offset)
            ys.append(value)
            lower.append(value - lo)
            upper.append(hi - value)
            point_labels.append(f"n={metric['n']}")
        ax.errorbar(
            xs,
            ys,
            yerr=[lower, upper],
            fmt="o",
            markersize=8,
            capsize=5,
            linewidth=2,
            color=MODEL_COLORS.get(model, "#475569"),
            label=model,
        )
        for x_value, y_value, label in zip(xs, ys, point_labels):
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#475569",
            )
    ax.set_xticks(range(len(tasks)), [TASK_LABELS.get(task, task) for task in tasks])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("内容准确率（95% Wilson CI）")
    _add_heading(
        ax,
        "分题型内容准确率",
        "仅统计可评分样本；n 为各模型实际可评分数量",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _finish_figure(fig, path)
    plt.close(fig)


def _plot_error_heatmap(path: Path, data: dict) -> None:
    import numpy as np

    plt = _pyplot()
    models = data["models"]
    error_names = ("wrong", "format_error", "truncated", "api_error")
    labels = ("内容错误", "格式错误", "截断", "API 错误")
    matrix = np.array(
        [[data["errors"][model][name] for name in error_names] for model in models]
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    image = ax.imshow(
        matrix,
        vmin=0,
        vmax=max(0.01, float(matrix.max())),
        cmap="YlOrRd",
        aspect="auto",
    )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(models)), models)
    for row in range(len(models)):
        for column in range(len(labels)):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > matrix.max() * 0.55 else "#0F172A",
                weight="bold",
            )
    ax.set_title("错误类型热力图（严格正式运行）", loc="left", weight="bold")
    fig.colorbar(image, ax=ax, shrink=0.8, label="占模型全部执行的比例")
    _finish_figure(fig, path)
    plt.close(fig)


def _plot_recovery_outcomes(path: Path, data: dict) -> None:
    import numpy as np

    plt = _pyplot()
    models = data["models"]
    status_order = ("correct", "wrong", "format_error", "truncated", "api_error")
    labels = {
        "correct": "恢复为正确",
        "wrong": "恢复为错误",
        "format_error": "转为格式错误",
        "truncated": "仍然截断",
        "api_error": "API 错误",
    }
    colors = {
        "correct": "#16A34A",
        "wrong": "#F59E0B",
        "format_error": "#7C3AED",
        "truncated": "#DC2626",
        "api_error": "#64748B",
    }
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for status in status_order:
        values = np.array(
            [data["recovery_outcomes"][model].get(status, 0) for model in models]
        )
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=labels[status],
            color=colors[status],
            width=0.58,
        )
        for bar, value, base in zip(bars, values, bottom):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                )
        bottom += values
    ax.set_xticks(x, models)
    ax.set_ylabel("截断样本数量")
    ax.set_title("39 条截断记录的扩展预算恢复结果", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _finish_figure(fig, path)
    plt.close(fig)


def generate_portfolio_figures(
    output_dir: Path,
    formal_summary: dict,
    recovery_summary: dict,
) -> list[Path]:
    """Generate the four approved 300-DPI portfolio figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_portfolio_data(formal_summary, recovery_summary)
    paths = [output_dir / name for name in FIGURE_FILENAMES]
    _plot_scorecard(paths[0], data)
    _plot_task_accuracy(paths[1], data)
    _plot_error_heatmap(paths[2], data)
    _plot_recovery_outcomes(paths[3], data)
    return paths
