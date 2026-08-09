param(
    [ValidateSet(
        "test",
        "baseline",
        "final",
        "estimator",
        "topology",
        "sensor",
        "figures",
        "experiments",
        "runtime",
        "clean"
    )]
    [string]$Task = "test"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Ensure-Dependencies {
    python -c "import numpy" | Out-Null
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Host "+ python -m pip install -r requirements-lock.txt"
    python -m pip install -r requirements-lock.txt
    python -c "import numpy" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install project dependencies."
    }
}

Ensure-Dependencies

switch ($Task) {
    "test" { python main.py verify }
    "baseline" { python main.py baseline --seeds 20 --workers 4 }
    "final" { python main.py final-validation --seeds 200 --workers 4 }
    "estimator" { python main.py estimator --seeds 50 --workers 4 }
    "topology" { python main.py topology --seeds 50 --workers 4 }
    "sensor" { python main.py sensor --seeds 30 --workers 4 }
    "figures" {
        python main.py check-figures
        python main.py figures
    }
    "experiments" { python main.py full }
    "runtime" { python main.py runtime --repeats 10 }
    "clean" {
        Get-ChildItem outputs\tables -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne ".gitkeep" } | Remove-Item -Force
        Get-ChildItem outputs\figures -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne ".gitkeep" } | Remove-Item -Force
    }
}
