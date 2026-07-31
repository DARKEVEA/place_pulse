param(
    [switch]$Resume,
    [int]$MaxGpuUtilization = 30,
    [int]$MaxGpuMemoryMiB = 4096
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_calibration_pilot.ps1"
$parameters = @{
    ConfigRelativePath = "configs\calibration_scalar_pilot_cuda.yaml"
    RunName = "run_005_scalar_calibration_pilot"
    MaxGpuUtilization = $MaxGpuUtilization
    MaxGpuMemoryMiB = $MaxGpuMemoryMiB
}
if ($Resume) {
    $parameters["Resume"] = $true
}

& $runner @parameters
