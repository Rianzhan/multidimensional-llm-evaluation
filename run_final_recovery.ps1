param(
    [Parameter(Mandatory = $false)]
    [string]$SourceRun = "",

    [Parameter(Mandatory = $false)]
    [string]$PriorRecoveryRun = "",

    [Parameter(Mandatory = $false)]
    [string]$ResumeFrom = ""
)

$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectDir

function Resolve-PythonPath {
    $venvPython = Join-Path $projectDir ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }
    return (Get-Command python -ErrorAction Stop).Source
}

if ([string]::IsNullOrWhiteSpace($SourceRun)) {
    $latestFormal = Get-ChildItem -LiteralPath (Join-Path $projectDir "runs") -Directory |
        Where-Object { $_.Name -like "formal-*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latestFormal) {
        throw "No formal run was found. Pass -SourceRun explicitly."
    }
    $SourceRun = $latestFormal.FullName
}
else {
    $SourceRun = (Resolve-Path -LiteralPath $SourceRun).Path
}

if ([string]::IsNullOrWhiteSpace($PriorRecoveryRun)) {
    $latestRecovery = Get-ChildItem -LiteralPath (Join-Path $projectDir "runs") -Directory |
        Where-Object {
            $_.Name -like "recovery-*" -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "recovery_results.json"))
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latestRecovery) {
        throw "No prior recovery run was found. Pass -PriorRecoveryRun explicitly."
    }
    $PriorRecoveryRun = $latestRecovery.FullName
}
else {
    $PriorRecoveryRun = (Resolve-Path -LiteralPath $PriorRecoveryRun).Path
}

if ([string]::IsNullOrWhiteSpace($ResumeFrom)) {
    $runId = Get-Date -Format "yyyyMMdd-HHmmss"
    $outputDir = "runs\final-recovery-$runId"
    $resumeArguments = @()
}
else {
    $outputDir = $ResumeFrom
    $resumeArguments = @("--resume")
}

function Read-ApiKey {
    param([Parameter(Mandatory = $true)][string]$Provider)

    $secureValue = Read-Host "$Provider API Key (input hidden)" -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
    try {
        $plainValue = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
        if ([string]::IsNullOrWhiteSpace($plainValue)) {
            throw "$Provider API Key cannot be empty"
        }
        return $plainValue
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

try {
    $env:GLM_API_KEY = Read-ApiKey -Provider "GLM"
    $env:KIMI_API_KEY = Read-ApiKey -Provider "KIMI"

    $pythonPath = Resolve-PythonPath
    $arguments = @(
        "recover.py",
        "--source-run",
        $SourceRun,
        "--prior-recovery-run",
        $PriorRecoveryRun,
        "--multiple-choice-tokens",
        "8192",
        "--output-dir",
        $outputDir
    )
    $arguments += $resumeArguments

    Write-Host "Formal source: $SourceRun"
    Write-Host "Prior recovery: $PriorRecoveryRun"
    Write-Host "Final retry output: $outputDir"
    Write-Host "Multiple-choice token budget: 8192"
    & $pythonPath @arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Final recovery failed with exit code: $LASTEXITCODE"
    }

    Write-Host "Final recovery results: $outputDir"
}
finally {
    Remove-Item Env:\GLM_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:\KIMI_API_KEY -ErrorAction SilentlyContinue
}
