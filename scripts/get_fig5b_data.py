#!/usr/bin/env python3
"""Create End Matter optimized-POVM comparison data.

This table summarizes the selected 4-QAM, 32-outcome rank-one POVM result and
the corresponding SRM baselines at the same channel loss.  The optimization is
a nonconvex multi-start Stiefel-manifold search, so the saved value is a
best-found lower bound rather than a proof of global optimality.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manuscript_reproduction.common import ordered_columns, parser, read_csv, write_csv


def apply_selected_povm_metadata(selected):
    """Use the saved POVM file as the canonical record for the selected point."""

    selected = selected.copy()
    for idx, row in selected.iterrows():
        matrix_path = ROOT / str(row["best_matrix_path_in_release"])
        matrix = np.load(matrix_path)
        selected.loc[idx, "M"] = int(matrix["M"])
        selected.loc[idx, "loss_db_per_arm"] = float(matrix["loss_db"])
        selected.loc[idx, "end_to_end_loss_db"] = 2.0 * float(matrix["loss_db"])
        selected.loc[idx, "spacing_d"] = float(matrix["spacing_d"])
        selected.loc[idx, "spacing_scale"] = float(matrix["spacing_scale"])
        selected.loc[idx, "optimized_32outcome_povm_rate"] = float(matrix["best_rate"])
    return selected


def main() -> None:
    args = parser(__doc__).parse_args()
    selected = read_csv(args.data_root, "optimized_povm/qam4_selected_32outcome_povm_comparison.csv")
    selected = apply_selected_povm_metadata(selected)
    selected["subfigure"] = "End Matter optimized-POVM comparison"
    selected["optimized_minus_raw_srm_at_same_d"] = (
        selected["optimized_32outcome_povm_rate"] - selected["raw_label_srm_rate_at_same_d"]
    )
    selected["relative_gain_vs_raw_srm_at_same_d_percent"] = (
        100.0
        * selected["optimized_minus_raw_srm_at_same_d"]
        / selected["raw_label_srm_rate_at_same_d"]
    )
    selected = ordered_columns(
        selected,
        [
            "subfigure",
            "M",
            "loss_db_per_arm",
            "end_to_end_loss_db",
            "spacing_d",
            "optimized_32outcome_povm_rate",
            "raw_label_srm_rate_at_same_d",
            "optimized_minus_raw_srm_at_same_d",
            "relative_gain_vs_raw_srm_at_same_d_percent",
            "raw_label_srm_optimized_rate_same_loss",
            "raw_label_srm_optimized_d_same_loss",
            "best_matrix_path_in_release",
        ],
    )

    scan = read_csv(args.data_root, "optimized_povm/best_by_d_scale.csv")
    scan["subfigure"] = "End Matter optimized-POVM d-scale scan"
    scan = ordered_columns(
        scan.sort_values("spacing_scale").reset_index(drop=True),
        [
            "subfigure",
            "spacing_scale",
            "M",
            "loss_db",
            "spacing_d",
            "outcomes",
            "previous_ykl_rate",
            "best_rate",
            "best_matrix_path",
            "source_file",
        ],
    )

    path1 = write_csv(selected, args.outdir, "fig5b_selected_32outcome_povm_comparison.csv")
    path2 = write_csv(scan, args.outdir, "fig5b_32outcome_povm_d_scale_scan.csv")
    print(path1)
    print(path2)


if __name__ == "__main__":
    main()
