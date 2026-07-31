param(
    [switch]$Resume,
    [int]$MaxGpuUtilization = 30,
    [int]$MaxGpuMemoryMiB = 4096
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_calibration_pilot.ps1"
$parameters = @{
    ConfigRelativePath = "configs\calibration_mixture_aggregation_pilot_cuda.yaml"
    RunName = "run_009_mixture_aggregation_pilot"
    MaxGpuUtilization = $MaxGpuUtilization
    MaxGpuMemoryMiB = $MaxGpuMemoryMiB
}
if ($Resume) {
    $parameters["Resume"] = $true
}

& $runner @parameters
