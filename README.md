# Event Egress Simulator

A Python project for simulating event egress on a mesoscopic queue network and evaluating dynamic route-guidance policies under uncertain capacity and partial compliance.

## Features

- discrete-time conservation-based queue network;
- source release, path queues, travel-delay cells, and exit queues;
- persistent stochastic path and exit capacities;
- partial compliance with route recommendations;
- static, reactive, scenario-mean, conditioned, CVaR, oracle, and fallback policies;
- paired-seed experiment runners and bootstrap summaries;
- CSV result export and generic Matplotlib figures;
- unit tests for conservation, normalization, policy behavior, and output statistics.

## Project layout

```text
config/                 Model and controller configuration
src/egress_sim/       Simulator, policies, uncertainty, analysis, plotting
experiments/            Standalone experiment runners
scripts/verify.py       Unit tests and a smoke simulation
tests/                  Core model and output-statistics tests
outputs/tables/         Generated CSV files
outputs/figures/        Generated plots
main.py                 Small command-line launcher
```

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Verify the project

```bash
python main.py verify
```

This runs the unit tests and one deterministic smoke simulation.

## Run a baseline experiment

```bash
python main.py baseline --seeds 20 --workers 4
```

Equivalent direct command:

```bash
python experiments/run_baselines.py --seeds 20 --workers 4
```

## Run the complete experiment suite

```bash
python main.py full
```

The complete suite is computationally intensive. Individual runners in `experiments/` accept their own `--seeds` and `--workers` options.

Examples:

```bash
python experiments/run_information_value_validation.py --seeds 200 --workers 8
python experiments/run_time_step_sensitivity.py --seeds 50 --workers 4
python experiments/run_sensor_ablation.py --seeds 30 --workers 4
python experiments/run_runtime_benchmark.py
```

## Configuration

The default network and controller settings are in `config/default.json`. The configuration specifies:

- zone populations and release rates;
- path and exit capacities;
- free-flow travel times;
- queue holding areas;
- capacity uncertainty and compliance;
- controller horizon, scenario count, candidate count, and objective weights.

## Outputs

Experiment runners create files under:

```text
outputs/tables/
outputs/figures/
```

These directories are intentionally empty in the distributed project.

## IEEE-style figures

The compact double-column figure set is generated entirely in Python.
The renderer reads CSV result tables and writes both vector PDF and 600-dpi PNG files.

Generate all required result tables with the validation settings and then render:

```bash
python main.py figure-pipeline --workers 8
```

This produces:

```text
outputs/figures/ieee/fig1_system_overview.pdf
outputs/figures/ieee/fig1_system_overview.png
outputs/figures/ieee/fig2_performance_landscape.pdf
outputs/figures/ieee/fig2_performance_landscape.png
outputs/figures/ieee/fig3_context_validation.pdf
outputs/figures/ieee/fig3_context_validation.png
```

To render from CSV tables that already exist:

```bash
python main.py figures
```

The style module is `src/egress_sim/ieee_style.py`. It fixes the IEEE double-column width at 7.16 inches, uses compact serif typography and STIX mathematics, embeds TrueType fonts in PDF, and exports PNG at 600 dpi. The actual multi-panel layouts are defined in `experiments/render_ieee_figures.py`.
