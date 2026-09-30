import importlib
from pathlib import Path

import pytest


def portfolio_module():
    try:
        return importlib.import_module("llm_eval.portfolio_figures")
    except ModuleNotFoundError:
        pytest.fail("llm_eval.portfolio_figures is not implemented")


def fixture_summaries():
    formal = {
        "by_model": {
            "GLM": {"total": 4},
            "KIMI": {"total": 4},
        },
        "by_task_type": {
            "multiple_choice": {
                "GLM": {
                    "content_accuracy": 0.75,
                    "content_accuracy_ci95": [0.30, 0.95],
                    "content_evaluable": 4,
                },
                "KIMI": {
                    "content_accuracy": 0.50,
                    "content_accuracy_ci95": [0.15, 0.85],
                    "content_evaluable": 2,
                },
            }
        },
    }
    recovery = {
        "by_model": {
            "GLM": {
                "total_results": 4,
                "strict_end_to_end_accuracy": 0.50,
                "effective_end_to_end_accuracy": 0.75,
                "strict_status_counts": {
                    "correct": 2,
                    "wrong": 1,
                    "truncated": 1,
                },
            },
            "KIMI": {
                "total_results": 4,
                "strict_end_to_end_accuracy": 0.25,
                "effective_end_to_end_accuracy": 0.25,
                "strict_status_counts": {
                    "correct": 1,
                    "wrong": 1,
                    "format_error": 1,
                    "truncated": 1,
                },
            },
        },
        "records": [
            {
                "model_name": "GLM",
                "recovery_status": "correct",
            },
            {
                "model_name": "KIMI",
                "recovery_status": "format_error",
            },
        ],
    }
    return formal, recovery


def test_prepare_portfolio_data_keeps_metric_denominators_explicit():
    formal, recovery = fixture_summaries()

    data = portfolio_module().prepare_portfolio_data(formal, recovery)

    assert data["models"] == ["GLM", "KIMI"]
    assert data["overview"]["GLM"] == {"strict": 0.5, "recovered": 0.75}
    assert data["tasks"]["multiple_choice"]["KIMI"] == {
        "accuracy": 0.5,
        "ci95": [0.15, 0.85],
        "n": 2,
    }
    assert data["errors"]["KIMI"] == {
        "wrong": 0.25,
        "format_error": 0.25,
        "truncated": 0.25,
        "api_error": 0.0,
    }
    assert data["recovery_outcomes"]["GLM"]["correct"] == 1
    assert data["recovery_outcomes"]["KIMI"]["format_error"] == 1


def test_generate_portfolio_figures_writes_four_high_resolution_pngs(tmp_path):
    formal, recovery = fixture_summaries()

    module = portfolio_module()
    paths = module.generate_portfolio_figures(tmp_path, formal, recovery)

    assert {path.name for path in paths} == set(module.FIGURE_FILENAMES)
    for path in paths:
        assert path.read_bytes().startswith(b"\x89PNG")
        assert path.stat().st_size > 1000


def test_chart_title_and_subtitle_do_not_overlap():
    module = portfolio_module()
    plt = module._pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))

    subtitle = module._add_heading(ax, "主标题", "解释性副标题")
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    assert not ax.title.get_window_extent(renderer).overlaps(
        subtitle.get_window_extent(renderer)
    )
    plt.close(fig)
