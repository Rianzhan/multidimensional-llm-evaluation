import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_entrypoint_can_validate_the_frozen_suite_without_network():
    """Catch packaging/path regressions that break a fresh-clone dry run."""

    environment = {
        **os.environ,
        "GLM_API_KEY": "test-only-glm-key",
        "KIMI_API_KEY": "test-only-kimi-key",
        "DEEPSEEK_API_KEY": "test-only-deepseek-key",
    }
    completed = subprocess.run(
        [sys.executable, "evaluate.py", "--dry-run"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "基础题: 90" in completed.stdout
    assert "稳定性请求: 135" in completed.stdout
    assert "dry-run 完成：未发送任何 API 请求。" in completed.stdout
