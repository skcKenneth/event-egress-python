from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egress_sim.figure_protocol import validate_frozen_artifact_bundle


def main() -> None:
    artifacts = ROOT / "paper_artifacts"
    validate_frozen_artifact_bundle(artifacts)
    print(
        "Frozen paper tables, figures, summaries, and SHA-256 manifests passed "
        f"validation: {artifacts}"
    )


if __name__ == "__main__":
    main()
