param(
    [switch]$Resume,
    [int]$MaxGpuUtilization = 30,
    [int]$MaxGpuMemoryMiB = 4096
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_calibration_pilot.ps1"
$parameters = @{
    ConfigRelativePath = "configs\calibration_multiseed_screening_cuda.yaml"
    RunName = "run_010_multiseed_model_screening"
    MaxGpuUtilization = $MaxGpuUtilization
    MaxGpuMemoryMiB = $MaxGpuMemoryMiB
}
if ($Resume) {
    $parameters["Resume"] = $true
}

& $runner @parameters
