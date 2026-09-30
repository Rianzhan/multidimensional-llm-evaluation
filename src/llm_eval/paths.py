"""Resolve repository-owned data without assuming an editable install."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


PROJECT_ROOT_ENV = "LLM_EVAL_PROJECT_ROOT"


def _has_project_assets(path: Path) -> bool:
    return (path / "config" / "models.json").is_file() and (path / "data").is_dir()


def resolve_project_root(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    source_checkout: Path | None = None,
) -> Path:
    """Return the checkout that owns config/data used by the CLI.

    An explicit environment override is authoritative. Otherwise the current
    checkout is preferred, followed by the source-layout location used during
    editable development installs.
    """

    environment = os.environ if environ is None else environ
    override = environment.get(PROJECT_ROOT_ENV, "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if not _has_project_assets(candidate):
            raise ValueError(
                f"{PROJECT_ROOT_ENV} does not point to a project checkout: {candidate}"
            )
        return candidate

    current = (Path.cwd() if cwd is None else cwd).resolve()
    source = (
        Path(__file__).resolve().parents[2]
        if source_checkout is None
        else source_checkout.resolve()
    )
    for candidate in (current, source):
        if _has_project_assets(candidate):
            return candidate

    raise RuntimeError(
        "Cannot locate config/models.json and data/. Run from the repository "
        f"root or set {PROJECT_ROOT_ENV}."
    )
