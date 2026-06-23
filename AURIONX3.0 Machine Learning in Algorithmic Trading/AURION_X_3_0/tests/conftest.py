"""Asegura imports de paquetes (`core`, `ml`, …) al ejecutar pytest desde la raíz del proyecto."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
