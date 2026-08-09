"""Large-event crowd-egress simulation package."""

from .config import EgressSpec, load_spec
from .simulation import SimulationResult, run_simulation

__version__ = "0.2.0"

__all__ = [
    "EgressSpec",
    "SimulationResult",
    "__version__",
    "load_spec",
    "run_simulation",
]
