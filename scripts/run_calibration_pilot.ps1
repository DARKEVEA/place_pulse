param(
    [switch]$Resume,
    [int]$MaxGpuUtilization = 30,
    [int]$MaxGpuMemoryMiB = 4096
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $repoRoot "configs\calibration_pilot_cuda.yaml"
$artifactRoot = Join-Path $repoRoot "artifacts\run_004_null_rule_pilot"
$logPath = Join-Path $repoRoot "run_004_null_rule_pilot.log"
$summaryPath = Join-Path $repoRoot "run_004_null_rule_pilot_summary.json"

if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Pilot config not found: $configPath"
}

if ($env:CONDA_DEFAULT_ENV -ne "arch") {
    throw "Activate the arch environment first: conda activate arch"
}

$ppc = Get-Command ppc -ErrorAction Stop
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction Stop
$gpuRaw = & $nvidiaSmi.Source `
    --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
    --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or -not $gpuRaw) {
    throw "Unable to query the NVIDIA GPU."
}

$gpuFields = ($gpuRaw | Select-Object -First 1).Split(",").Trim()
$gpuUtilization = [int]$gpuFields[0]
$gpuMemoryUsed = [int]$gpuFields[1]
$gpuMemoryTotal = [int]$gpuFields[2]
$gpuTemperature = [int]$gpuFields[3]

Write-Host (
    "GPU before pilot: utilization={0}% memory={1}/{2} MiB temperature={3}C" -f `
        $gpuUtilization, $gpuMemoryUsed, $gpuMemoryTotal, $gpuTemperature
)

if (
    $gpuUtilization -gt $MaxGpuUtilization -or
    $gpuMemoryUsed -gt $MaxGpuMemoryMiB
) {
    $busyMessage = (
        "GPU is busy; pilot was not started. Limits: utilization <= {0}%, " +
        "memory <= {1} MiB."
    ) -f $MaxGpuUtilization, $MaxGpuMemoryMiB
    throw $busyMessage
}

$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$arguments = @(
    "simulate",
    "validate-models",
    "--config",
    $configPath
)
if ($Resume) {
    $arguments += "--resume"
}

New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$startedAt = Get-Date
Write-Host "Pilot started at $($startedAt.ToString('o'))"
Write-Host "Command: ppc $($arguments -join ' ')"

# Windows PowerShell wraps native stderr records as NativeCommandError. Keep
# streaming them to the log without treating ordinary progress output as a
# terminating PowerShell exception.
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
    config = $configPath
    artifacts = $artifactRoot
    log = $logPath
}
$summary | ConvertTo-Json | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$summary | Format-List

if ($exitCode -ne 0) {
    throw "Calibration pilot failed with exit code $exitCode. See $logPath"
}
