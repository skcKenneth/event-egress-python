param(
    [ValidateSet("test", "baseline", "experiments", "runtime", "clean")]
    [string]$Task = "test"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

switch ($Task) {
    "test" { python main.py verify }
    "baseline" { python main.py baseline --seeds 20 --workers 4 }
    "experiments" { python main.py full }
    "runtime" { python main.py runtime --repeats 10 }
    "clean" {
        Get-ChildItem outputs\tables -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne ".gitkeep" } | Remove-Item -Force
        Get-ChildItem outputs\figures -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne ".gitkeep" } | Remove-Item -Force
    }
}
