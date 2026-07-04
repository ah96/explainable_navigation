"""Make the package importable when running pytest from a fresh checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
