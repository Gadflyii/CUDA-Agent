#!/usr/bin/env python3
"""Run the full reproducible intake, normalization, export, and validation pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEPS = ["collect_ginfer.py", "collect_external.py", "fetch_public.py", "build_normalized.py", "export_dataset.py", "validate.py"]


def main() -> None:
    for script in STEPS:
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
