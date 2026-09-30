$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectDir
$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$outputDir = "results_v2\smoke-$runId"

function Resolve-PythonPath {
    $venvPython = Join-Path $projectDir ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }
    return (Get-Command python -ErrorAction Stop).Source
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
    $env:DEEPSEEK_API_KEY = Read-ApiKey -Provider "DEEPSEEK"

    $pythonPath = Resolve-PythonPath
    & $pythonPath evaluate.py `
        --dataset data\smoke_3.jsonl `
        --skip-stability `
        --output-dir $outputDir

    if ($LASTEXITCODE -ne 0) {
        throw "Smoke test failed with exit code: $LASTEXITCODE"
    }

    Write-Host "Smoke results: $outputDir"
}
finally {
    Remove-Item Env:\GLM_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:\KIMI_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:\DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
}
