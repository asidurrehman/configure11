"""Shared pytest set-up for the CONFIGURE v1.1 tests."""
import sys
from pathlib import Path

import matplotlib

# Draw figures off-screen so the tests also run on machines without a display.
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
