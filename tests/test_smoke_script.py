import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = PROJECT_ROOT / "run_smoke.ps1"
FORMAL_SCRIPT = PROJECT_ROOT / "run_formal.ps1"
RECOVERY_SCRIPT = PROJECT_ROOT / "run_recovery.ps1"
FINAL_RECOVERY_SCRIPT = PROJECT_ROOT / "run_final_recovery.ps1"


def test_smoke_script_is_ascii_for_windows_powershell_compatibility():
    SMOKE_SCRIPT.read_bytes().decode("ascii")


def test_smoke_script_uses_a_unique_output_directory_per_run():
    script = SMOKE_SCRIPT.read_text(encoding="ascii")

    assert 'Get-Date -Format "yyyyMMdd-HHmmss"' in script
    assert '--output-dir $outputDir' in script


def test_formal_script_is_ascii_and_runs_all_evaluation_dimensions():
    script = FORMAL_SCRIPT.read_bytes().decode("ascii")

    assert "--run-robustness" in script
    assert 'Get-Date -Format "yyyyMMdd-HHmmss"' in script
    assert '"--resume"' in script


def test_recovery_script_is_ascii_for_windows_powershell_compatibility():
    RECOVERY_SCRIPT.read_bytes().decode("ascii")


def test_final_recovery_script_is_ascii_for_windows_powershell_compatibility():
    FINAL_RECOVERY_SCRIPT.read_bytes().decode("ascii")


@pytest.mark.skipif(
    shutil.which("powershell.exe") is None,
    reason="Windows PowerShell is not installed",
)
def test_windows_powershell_51_can_parse_smoke_script():
    parser_command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "(Resolve-Path '.\\run_smoke.ps1'), [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            parser_command,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(
    shutil.which("powershell.exe") is None,
    reason="Windows PowerShell is not installed",
)
def test_windows_powershell_51_can_parse_formal_script():
    parser_command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "(Resolve-Path '.\\run_formal.ps1'), [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            parser_command,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(
    shutil.which("powershell.exe") is None,
    reason="Windows PowerShell is not installed",
)
def test_windows_powershell_51_can_parse_recovery_script():
    parser_command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "(Resolve-Path '.\\run_recovery.ps1'), [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            parser_command,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(
    shutil.which("powershell.exe") is None,
    reason="Windows PowerShell is not installed",
)
def test_windows_powershell_51_can_parse_final_recovery_script():
    parser_command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "(Resolve-Path '.\\run_final_recovery.ps1'), [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            parser_command,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
