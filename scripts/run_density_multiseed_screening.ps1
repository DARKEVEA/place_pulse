param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_density_calibration_pilot.ps1"
$parameters = @{
    ConfigRelativePath = "configs\calibration_density_multiseed_cuda.yaml"
    RunName = "run_012_density_multiseed_screening"
}
if ($Resume) {
    $parameters["Resume"] = $true
}

& $runner @parameters
