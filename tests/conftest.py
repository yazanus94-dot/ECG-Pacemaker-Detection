"""Shared pytest configuration.

Adds the repository root to ``sys.path`` so ``src`` is importable as a package
regardless of the directory pytest is invoked from.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
