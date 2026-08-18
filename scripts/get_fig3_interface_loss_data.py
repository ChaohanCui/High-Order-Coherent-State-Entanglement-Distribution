#!/usr/bin/env python3
"""Create the interface-loss data table for Fig. 3(a,b).

The simulation models the repeated-reflection source/interface loss.  The
manuscript setting ``L_ch = 0.5 dB`` corresponds to ``0.25 dB`` per arm in the
code.  The exported table contains the full-state/raw-label SRM global optimum
for each interface loss value.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manuscript_reproduction.common import ordered_columns, parser, read_csv, write_csv


def main() -> None:
    args = parser(__doc__).parse_args()
    data = read_csv(
        args.data_root,
        "interface_loss/reflection_source_raw_label_global_optima_with_ultradense16.csv",
    )
    data["subfigure"] = "Fig. 3(a,b)"
    data["interface_loss_db_per_reflection"] = data["generation_loss_db_per_step"].astype(float)
    data["channel_loss_db_per_arm_code"] = data["channel_loss_db"].astype(float)
    data["end_to_end_channel_loss_db_midpoint"] = 2.0 * data["channel_loss_db_per_arm_code"]
    data = ordered_columns(
        data.sort_values(["M", "interface_loss_db_per_reflection"]).reset_index(drop=True),
        [
            "subfigure",
            "M",
            "constellation",
            "receiver",
            "interface_loss_db_per_reflection",
            "channel_loss_db_per_arm_code",
            "end_to_end_channel_loss_db_midpoint",
            "branch_label",
            "spacing_d",
            "hashing_bound_bits_per_attempt",
        ],
    )
    path = write_csv(data, args.outdir, "fig3ab_interface_loss_raw_label_srm.csv")
    print(path)


if __name__ == "__main__":
    main()

