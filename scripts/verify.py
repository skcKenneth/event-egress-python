from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egress_sim.config import load_spec
from egress_sim.policies import StaticNearestPolicy
from egress_sim.simulation import run_simulation


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])

    result = run_simulation(
        load_spec(ROOT / "config" / "default.json"),
        StaticNearestPolicy(),
        seed=101,
    )
    if not result.completed:
        raise SystemExit("Smoke simulation did not clear within the maximum duration.")
    if result.mass_balance_error > 1e-7:
        raise SystemExit(
            f"Smoke simulation mass-balance error is too large: "
            f"{result.mass_balance_error:.3e}"
        )

    print(
        "Verification passed: "
        f"clearance={result.clearance_time_seconds:.1f}s, "
        f"mass_error={result.mass_balance_error:.3e}"
    )


if __name__ == "__main__":
    main()
