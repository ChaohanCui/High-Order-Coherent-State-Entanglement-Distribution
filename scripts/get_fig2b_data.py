#!/usr/bin/env python3
"""Create the fixed-loss hashing-bound-versus-d tables for Fig. 2(b).

The raw input is a dense sweep of the QAM spacing ``d`` for 4-, 8-, and 16-QAM
at channel losses 0.9 and 0.95 dB per arm.  The local-maxima table is exported
beside the sweep table because it marks the branch switch.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manuscript_reproduction.common import ordered_columns, parser, read_csv, write_csv


def main() -> None:
    args = parser(__doc__).parse_args()
    sweep = read_csv(data_root=args.data_root, relative_path="d_sweep/raw_label_srm_qam_branch_sweep_points.csv")
    sweep["subfigure"] = "Fig. 2(b)"
    sweep["receiver"] = "full_state_raw_label_srm"
    sweep["loss_db_per_arm_code"] = sweep["loss_db"].astype(float)
    sweep["end_to_end_loss_db_midpoint"] = 2.0 * sweep["loss_db_per_arm_code"]
    sweep = ordered_columns(
        sweep.sort_values(["M", "loss_db", "spacing_d"]).reset_index(drop=True),
        [
            "subfigure",
            "M",
            "constellation",
            "receiver",
            "loss_db_per_arm_code",
            "end_to_end_loss_db_midpoint",
            "spacing_d",
            "hashing_bound_bits_per_attempt",
            "grid_region",
        ],
    )

    maxima = read_csv(data_root=args.data_root, relative_path="d_sweep/raw_label_srm_qam_branch_sweep_local_maxima.csv")
    maxima["subfigure"] = "Fig. 2(b)"
    maxima["receiver"] = "full_state_raw_label_srm"

    path1 = write_csv(sweep, args.outdir, "fig2b_hashing_bound_vs_d_sweep.csv")
    path2 = write_csv(maxima, args.outdir, "fig2b_local_maxima.csv")
    print(path1)
    print(path2)


if __name__ == "__main__":
    main()

