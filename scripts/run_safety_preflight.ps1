param(
    [switch]$Resume,
    [int]$MaxGpuUtilization = 30,
    [int]$MaxGpuMemoryMiB = 4096
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $repoRoot "configs\safety_provisional_preflight_cuda.yaml"
$logPath = Join-Path $repoRoot "run_018_safety_preflight.log"
$summaryPath = Join-Path $repoRoot "run_018_safety_preflight_summary.json"

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
    "GPU before preflight: utilization={0}% memory={1}/{2} MiB temperature={3}C" -f `
        $gpuUtilization, $gpuMemoryUsed, $gpuMemoryTotal, $gpuTemperature
)
if ($gpuUtilization -gt $MaxGpuUtilization -or $gpuMemoryUsed -gt $MaxGpuMemoryMiB) {
    throw "GPU is busy; Safety preflight was not started."
}

$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$arguments = @("run", "heterogeneity", "--config", $configPath)
if ($Resume) {
    $arguments += "--resume"
}

$startedAt = Get-Date
Write-Host "Safety preflight started at $($startedAt.ToString('o'))"
Write-Host "Command: ppc $($arguments -join ' ')"
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
    config = $configPath
    artifacts = Join-Path $repoRoot "artifacts\run_018_safety_preflight"
    log = $logPath
}
$summary | ConvertTo-Json | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$summary | Format-List
if ($exitCode -ne 0) {
    throw "Safety preflight failed with exit code $exitCode. See $logPath"
}
