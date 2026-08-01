param(
    [switch]$Resume,
    [int]$MaxGpuUtilization = 30,
    [int]$MaxGpuMemoryMiB = 4096
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_calibration_pilot.ps1"
$parameters = @{
    ConfigRelativePath = "configs\calibration_boundary_relevance_pilot_cuda.yaml"
    RunName = "run_016_boundary_relevance_pilot"
    MaxGpuUtilization = $MaxGpuUtilization
    MaxGpuMemoryMiB = $MaxGpuMemoryMiB
}
if ($Resume) {
    $parameters["Resume"] = $true
}

& $runner @parameters
