param(
    [switch]$Resume,
    [int]$MaxGpuUtilization = 30,
    [int]$MaxGpuMemoryMiB = 4096,
    [string]$ConfigRelativePath = "configs\calibration_pilot_cuda.yaml",
    [string]$RunName = "run_004_null_rule_pilot"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedConfigPath = if ([System.IO.Path]::IsPathRooted($ConfigRelativePath)) {
    $ConfigRelativePath
} else {
    Join-Path $repoRoot $ConfigRelativePath
}
$artifactRoot = Join-Path $repoRoot "artifacts\$RunName"
$logPath = Join-Path $repoRoot "$RunName.log"
$summaryPath = Join-Path $repoRoot "${RunName}_summary.json"

if (-not (Test-Path -LiteralPath $resolvedConfigPath)) {
    throw "Pilot config not found: $resolvedConfigPath"
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
    $resolvedConfigPath
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
    config = $resolvedConfigPath
    artifacts = $artifactRoot
    log = $logPath
}
$summary | ConvertTo-Json | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$summary | Format-List

if ($exitCode -ne 0) {
    throw "Calibration pilot failed with exit code $exitCode. See $logPath"
}
