param(
    [switch]$Resume,
    [int[]]$Seeds = @(1103, 2207, 3319, 4421, 5527),
    [int]$MaxGpuUtilization = 30,
    [int]$MaxGpuMemoryMiB = 6144
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $repoRoot "artifacts\run_021_wealthy_provisional_multiseed"
$summaryPath = Join-Path $repoRoot "run_021_wealthy_provisional_multiseed_summary.json"
$allowedSeeds = @(1103, 2207, 3319, 4421, 5527)

if ($env:CONDA_DEFAULT_ENV -ne "arch") {
    throw "Activate the arch environment first: conda activate arch"
}
if (-not $Seeds -or $Seeds.Count -eq 0) {
    throw "At least one seed is required."
}
foreach ($seed in $Seeds) {
    if ($seed -notin $allowedSeeds) {
        throw "Unsupported seed $seed. Allowed seeds: $($allowedSeeds -join ', ')"
    }
}

$ppc = Get-Command ppc -ErrorAction Stop
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction Stop
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$runStartedAt = Get-Date
$seedSummaries = @()
foreach ($seed in $Seeds) {
    $configPath = Join-Path $repoRoot "configs\wealthy_provisional_seeds\seed_$seed.yaml"
    $artifactRoot = Join-Path $runRoot "seed_$seed"
    $logPath = Join-Path $repoRoot "run_021_wealthy_seed_$seed.log"
    if (-not (Test-Path -LiteralPath $configPath)) {
        throw "Seed config not found: $configPath"
    }

    $gpuRaw = & $nvidiaSmi.Source `
        --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
        --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0 -or -not $gpuRaw) {
        throw "Unable to query the NVIDIA GPU before seed $seed."
    }
    $gpuFields = ($gpuRaw | Select-Object -First 1).Split(",").Trim()
    $gpuUtilization = [int]$gpuFields[0]
    $gpuMemoryUsed = [int]$gpuFields[1]
    $gpuMemoryTotal = [int]$gpuFields[2]
    $gpuTemperature = [int]$gpuFields[3]
    Write-Host (
        "GPU before seed {0}: utilization={1}% memory={2}/{3} MiB temperature={4}C" -f `
            $seed, $gpuUtilization, $gpuMemoryUsed, $gpuMemoryTotal, $gpuTemperature
    )
    if (
        $gpuUtilization -gt $MaxGpuUtilization -or
        $gpuMemoryUsed -gt $MaxGpuMemoryMiB
    ) {
        throw (
            "GPU is busy; seed {0} was not started. Limits: utilization <= {1}%, " +
            "memory <= {2} MiB."
        ) -f $seed, $MaxGpuUtilization, $MaxGpuMemoryMiB
    }

    $arguments = @("run", "heterogeneity", "--config", $configPath)
    if ($Resume) {
        $arguments += "--resume"
    }
    $startedAt = Get-Date
    Write-Host "Wealthy seed $seed started at $($startedAt.ToString('o'))"
    Write-Host "Command: ppc $($arguments -join ' ')"
    $savedErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $ppc.Source @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorPreference
    $finishedAt = Get-Date
    $elapsed = $finishedAt - $startedAt
    $seedSummary = [ordered]@{
        dimension = "wealthy"
        seed = $seed
        status = if ($exitCode -eq 0) { "complete" } else { "failed" }
        exit_code = $exitCode
        provisional = $true
        confirmatory = $false
        replication_screening = $true
        resumed = [bool]$Resume
        started_at = $startedAt.ToString("o")
        finished_at = $finishedAt.ToString("o")
        elapsed_seconds = [math]::Round($elapsed.TotalSeconds, 3)
        elapsed_hours = [math]::Round($elapsed.TotalHours, 3)
        config = $configPath
        artifacts = $artifactRoot
        log = $logPath
    }
    $seedSummaries += $seedSummary
    $runSummary = [ordered]@{
        status = if ($exitCode -eq 0) { "running" } else { "failed" }
        dimension = "wealthy"
        provisional = $true
        confirmatory = $false
        replication_screening = $true
        calibration_policy = "provisional_effective"
        resumed = [bool]$Resume
        requested_seeds = $Seeds
        started_at = $runStartedAt.ToString("o")
        updated_at = $finishedAt.ToString("o")
        seeds = $seedSummaries
    }
    $runSummary | ConvertTo-Json -Depth 6 | Set-Content `
        -LiteralPath $summaryPath -Encoding UTF8
    $seedSummary | Format-List
    if ($exitCode -ne 0) {
        throw "Wealthy seed $seed failed with exit code $exitCode. See $logPath"
    }
}

$runFinishedAt = Get-Date
$finalSummary = [ordered]@{
    status = "complete"
    dimension = "wealthy"
    provisional = $true
    confirmatory = $false
    replication_screening = $true
    calibration_policy = "provisional_effective"
    resumed = [bool]$Resume
    requested_seeds = $Seeds
    started_at = $runStartedAt.ToString("o")
    finished_at = $runFinishedAt.ToString("o")
    elapsed_seconds = [math]::Round(($runFinishedAt - $runStartedAt).TotalSeconds, 3)
    elapsed_hours = [math]::Round(($runFinishedAt - $runStartedAt).TotalHours, 3)
    seeds = $seedSummaries
}
$finalSummary | ConvertTo-Json -Depth 6 | Set-Content `
    -LiteralPath $summaryPath -Encoding UTF8
$finalSummary | Format-List
