param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_density_calibration_pilot.ps1"
$parameters = @{
    ConfigRelativePath = "configs\calibration_density_diagnostics_cuda.yaml"
    RunName = "run_013_density_diagnostics"
}
if ($Resume) {
    $parameters["Resume"] = $true
}

& $runner @parameters
