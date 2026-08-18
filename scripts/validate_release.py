#!/usr/bin/env python3
"""Lightweight validation for the GitHub release folder.

The check verifies that required curated data files exist and that each
subfigure extraction script can run.  It intentionally avoids expensive
regeneration.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "data/raw/ideal_channel_srm/raw_label_vs_vacuum_omit_srm_all_data.csv",
    "data/raw/ideal_channel_srm/qam32_raw_label_vs_vacuum_merged.csv",
    "data/raw/d_sweep/raw_label_srm_qam_branch_sweep_points.csv",
    "data/raw/interface_loss/reflection_source_raw_label_global_optima_with_ultradense16.csv",
    "data/raw/phase_error/interface_0p1db/raw_label_phase_error_summary.csv",
    "data/raw/phase_error/interface_0p2db/raw_label_phase_error_summary.csv",
    "data/raw/optimized_povm/qam4_selected_32outcome_povm_comparison.csv",
    "data/raw/optimized_povm/best_M4_loss_0.25_scale_0p93_outcomes_32.npz",
    "source_code/src/qam_source_loss_hashing.py",
    "source_code/src/qam_reflection_source_loss_hashing.py",
]


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        for path in missing:
            print(f"missing: {path}")
        raise SystemExit(1)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_all_subfigure_data.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "spot_check_simulation_samples.py")], check=True)
    print("Release validation completed successfully.")


if __name__ == "__main__":
    main()
