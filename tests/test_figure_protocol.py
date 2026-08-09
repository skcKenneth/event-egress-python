from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egress_sim.figure_protocol import (
    FigureProtocolError,
    validate_frozen_artifact_bundle,
    validate_reference_figure_inputs,
)


class FigureProtocolTests(unittest.TestCase):
    def test_bundled_figure_tables_match_reference_protocol(self) -> None:
        validate_frozen_artifact_bundle(ROOT / "paper_artifacts")

    def test_modified_information_map_is_rejected(self) -> None:
        source_root = ROOT / "paper_artifacts"
        with tempfile.TemporaryDirectory() as directory:
            target_root = Path(directory)
            shutil.copytree(source_root / "tables", target_root / "tables")
            shutil.copy2(
                source_root / "reference_manifest.json",
                target_root / "reference_manifest.json",
            )
            summary_path = target_root / "tables" / "information_regime_map_summary.csv"
            summary = pd.read_csv(summary_path)
            summary.loc[0, "mean_difference"] += 1.0
            summary.to_csv(summary_path, index=False)
            with self.assertRaisesRegex(
                FigureProtocolError,
                "hash mismatch",
            ):
                validate_reference_figure_inputs(target_root / "tables")


if __name__ == "__main__":
    unittest.main()
