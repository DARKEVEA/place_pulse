param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $repoRoot "configs\calibration_density_pilot_cuda.yaml"
$artifactRoot = Join-Path $repoRoot "artifacts\run_011_density_calibration_pilot"
$logPath = Join-Path $repoRoot "run_011_density_calibration_pilot.log"
$summaryPath = Join-Path $repoRoot "run_011_density_calibration_pilot_summary.json"

if ($env:CONDA_DEFAULT_ENV -ne "arch") {
    throw "Activate the arch environment first: conda activate arch"
}

$ppc = Get-Command ppc -ErrorAction Stop
$arguments = @(
    "simulate",
    "validate-density",
    "--config",
    $configPath
)
if ($Resume) {
    $arguments += "--resume"
}

New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$startedAt = Get-Date
Write-Host "RUN_011 density pilot uses the CPU implementation; GPU availability is irrelevant."
Write-Host "Pilot started at $($startedAt.ToString('o'))"
Write-Host "Command: ppc $($arguments -join ' ')"

# Windows PowerShell wraps native stderr records as NativeCommandError. Stream
# progress to the log without treating ordinary status output as terminating.
$savedErrorPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $ppc.Source @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorPreference

$finishedAt = Get-Date
$elapsed = $finishedAt - $startedAt
$summary = [ordered]@{
    status = if ($exitCode -eq 0) { "complete" } else { "failed" }
    exit_code = $exitCode
    resumed = [bool]$Resume
    started_at = $startedAt.ToString("o")
    finished_at = $finishedAt.ToString("o")
    elapsed_seconds = [math]::Round($elapsed.TotalSeconds, 3)
    elapsed_minutes = [math]::Round($elapsed.TotalMinutes, 3)
    execution_backend = "cpu_scipy_numpy"
    config = $configPath
    artifacts = $artifactRoot
    log = $logPath
}
$summary | ConvertTo-Json | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$summary | Format-List

if ($exitCode -ne 0) {
    throw "Density pilot failed with exit code $exitCode. See $logPath"
}
