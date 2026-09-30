from pathlib import Path

from llm_eval.paths import resolve_project_root


def _make_project_root(path: Path) -> Path:
    (path / "config").mkdir(parents=True)
    (path / "data").mkdir()
    (path / "config" / "models.json").write_text("[]", encoding="utf-8")
    return path


def test_explicit_project_root_wins_over_the_current_directory(tmp_path):
    """Catch installed-CLI runs silently reading a different checkout."""

    explicit = _make_project_root(tmp_path / "explicit")
    current = _make_project_root(tmp_path / "current")

    resolved = resolve_project_root(
        environ={"LLM_EVAL_PROJECT_ROOT": str(explicit)},
        cwd=current,
        source_checkout=tmp_path / "missing-source",
    )

    assert resolved == explicit.resolve()


def test_current_checkout_is_used_when_no_override_is_set(tmp_path):
    """Catch wheel installs incorrectly treating site-packages as project data."""

    current = _make_project_root(tmp_path / "current")

    resolved = resolve_project_root(
        environ={},
        cwd=current,
        source_checkout=tmp_path / "site-packages",
    )

    assert resolved == current.resolve()
