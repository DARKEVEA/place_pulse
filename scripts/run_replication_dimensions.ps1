param(
    [switch]$Resume,
    [string[]]$Dimensions = @(
        "lively", "beautiful", "wealthy", "boring", "depressing"
    ),
    [int]$MaxGpuUtilization = 30,
    [int]$MaxGpuMemoryMiB = 6144
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $repoRoot "artifacts\run_020_replication_dimensions"
$overallSummaryPath = Join-Path $repoRoot "run_020_replication_dimensions_summary.json"
$allowedDimensions = @("lively", "beautiful", "wealthy", "boring", "depressing")

if ($env:CONDA_DEFAULT_ENV -ne "arch") {
    throw "Activate the arch environment first: conda activate arch"
}
if (-not $Dimensions -or $Dimensions.Count -eq 0) {
    throw "At least one replication dimension is required."
}
foreach ($dimension in $Dimensions) {
    if ($dimension -notin $allowedDimensions) {
        throw (
            "Unsupported dimension {0}. Allowed dimensions: {1}" -f `
                $dimension, ($allowedDimensions -join ", ")
        )
    }
}

$ppc = Get-Command ppc -ErrorAction Stop
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction Stop
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$invocationStartedAt = Get-Date

foreach ($dimension in $Dimensions) {
    $configPath = Join-Path $repoRoot "configs\replication_dimensions\$dimension.yaml"
    $artifactRoot = Join-Path $runRoot "${dimension}_seed_1103"
    $logPath = Join-Path $repoRoot "run_020_${dimension}_seed_1103.log"
    $dimensionSummaryPath = Join-Path `
        $repoRoot "run_020_${dimension}_seed_1103_summary.json"
    if (-not (Test-Path -LiteralPath $configPath)) {
        throw "Replication config not found: $configPath"
    }

    $gpuRaw = & $nvidiaSmi.Source `
        --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
        --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0 -or -not $gpuRaw) {
        throw "Unable to query the NVIDIA GPU before $dimension."
    }
    $gpuFields = ($gpuRaw | Select-Object -First 1).Split(",").Trim()
    $gpuUtilization = [int]$gpuFields[0]
    $gpuMemoryUsed = [int]$gpuFields[1]
    $gpuMemoryTotal = [int]$gpuFields[2]
    $gpuTemperature = [int]$gpuFields[3]
    Write-Host (
        "GPU before {0}: utilization={1}% memory={2}/{3} MiB temperature={4}C" -f `
            $dimension, $gpuUtilization, $gpuMemoryUsed, `
            $gpuMemoryTotal, $gpuTemperature
    )
    if (
        $gpuUtilization -gt $MaxGpuUtilization -or
        $gpuMemoryUsed -gt $MaxGpuMemoryMiB
    ) {
        throw (
            "GPU is busy; {0} was not started. Limits: utilization <= {1}%, " +
            "memory <= {2} MiB."
        ) -f $dimension, $MaxGpuUtilization, $MaxGpuMemoryMiB
    }

    $arguments = @("run", "heterogeneity", "--config", $configPath)
    if ($Resume) {
        $arguments += "--resume"
    }
    $startedAt = Get-Date
    Write-Host "$dimension started at $($startedAt.ToString('o'))"
    Write-Host "Command: ppc $($arguments -join ' ')"
    $savedErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $ppc.Source @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorPreference
    $finishedAt = Get-Date
    $elapsed = $finishedAt - $startedAt
    $dimensionSummary = [ordered]@{
        dimension = $dimension
        seed = 1103
        status = if ($exitCode -eq 0) { "complete" } else { "failed" }
        exit_code = $exitCode
        provisional = $true
        confirmatory = $false
        replication = $true
        resumed = [bool]$Resume
        started_at = $startedAt.ToString("o")
        finished_at = $finishedAt.ToString("o")
        elapsed_seconds = [math]::Round($elapsed.TotalSeconds, 3)
        elapsed_hours = [math]::Round($elapsed.TotalHours, 3)
        config = $configPath
        artifacts = $artifactRoot
        log = $logPath
    }
    $dimensionSummary | ConvertTo-Json -Depth 6 | Set-Content `
        -LiteralPath $dimensionSummaryPath -Encoding UTF8
    $dimensionSummary | Format-List

    $recorded = @()
    foreach ($knownDimension in $allowedDimensions) {
        $knownSummaryPath = Join-Path `
            $repoRoot "run_020_${knownDimension}_seed_1103_summary.json"
        if (Test-Path -LiteralPath $knownSummaryPath) {
            $recorded += Get-Content -LiteralPath $knownSummaryPath -Raw | ConvertFrom-Json
        }
    }
    $completeCount = @($recorded | Where-Object { $_.status -eq "complete" }).Count
    $overallSummary = [ordered]@{
        status = if ($completeCount -eq $allowedDimensions.Count) {
            "complete"
        } elseif ($exitCode -ne 0) {
            "failed"
        } else {
            "partial"
        }
        provisional = $true
        confirmatory = $false
        replication = $true
        seed = 1103
        requested_dimensions = $Dimensions
        completed_dimensions = $completeCount
        total_dimensions = $allowedDimensions.Count
        invocation_started_at = $invocationStartedAt.ToString("o")
        updated_at = $finishedAt.ToString("o")
        dimensions = $recorded
    }
    $overallSummary | ConvertTo-Json -Depth 8 | Set-Content `
        -LiteralPath $overallSummaryPath -Encoding UTF8
    if ($exitCode -ne 0) {
        throw "$dimension failed with exit code $exitCode. See $logPath"
    }
}

Get-Content -LiteralPath $overallSummaryPath -Raw | ConvertFrom-Json | Format-List
