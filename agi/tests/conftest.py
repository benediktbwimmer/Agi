from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable (for legacy imports)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure the src directory is importable when running tests directly from the repository
SRC_PATH = PROJECT_ROOT / "agi" / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))
