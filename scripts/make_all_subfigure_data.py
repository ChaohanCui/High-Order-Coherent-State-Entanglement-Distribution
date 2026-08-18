#!/usr/bin/env python3
"""Run every subfigure-data extraction script.

This is the quickest reproducibility entry point for reviewers: it does not
rerun expensive simulations, it converts the curated raw data in ``data/raw``
into small CSV tables under ``outputs/subfigure_data``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "get_fig1c_data.py",
    "get_fig2a_data.py",
    "get_fig2b_data.py",
    "get_fig3_interface_loss_data.py",
    "get_fig3_phase_bias_data.py",
    "get_fig5a_data.py",
    "get_fig5b_data.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"Running {script}")
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)
    print(f"Wrote subfigure CSVs to {ROOT / 'outputs' / 'subfigure_data'}")


if __name__ == "__main__":
    main()

