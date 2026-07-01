from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# One worker per physical core; cap at 16 for high thread-count CPUs.
DEFAULT_WORKERS = min(16, max(1, (os.cpu_count() or 8) // 2))

