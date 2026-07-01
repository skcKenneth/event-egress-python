from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from egress_sim.figure_protocol import FigureProtocolError, validate_reference_figure_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_figure_tables_match_reference_protocol() -> None:
    validate_reference_figure_inputs(ROOT / "outputs" / "tables")


def test_partial_information_map_is_rejected(tmp_path: Path) -> None:
    source = ROOT / "outputs" / "tables"
    for path in source.glob("*.csv"):
        (tmp_path / path.name).write_bytes(path.read_bytes())

    summary_path = tmp_path / "information_regime_map_summary.csv"
    summary = pd.read_csv(summary_path)
    summary["runs"] = 4
    summary.to_csv(summary_path, index=False)

    with pytest.raises(FigureProtocolError, match="50 runs"):
        validate_reference_figure_inputs(tmp_path)
