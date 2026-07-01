from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egress_sim.figure_protocol import validate_reference_figure_inputs


if __name__ == "__main__":
    tables = ROOT / "outputs" / "tables"
    validate_reference_figure_inputs(tables)
    print("IEEE figure inputs match the frozen reference protocol and values.")
