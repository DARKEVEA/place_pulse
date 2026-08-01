param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_density_calibration_pilot.ps1"
$parameters = @{
    ConfigRelativePath = "configs\calibration_density_confirmatory_cuda.yaml"
    RunName = "run_014_density_confirmatory"
}
if ($Resume) {
    $parameters["Resume"] = $true
}

& $runner @parameters
