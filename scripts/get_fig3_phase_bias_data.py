#!/usr/bin/env python3
"""Create the phase-bias data table for Fig. 3(c,d).

The phase-bias scan sweeps the systematic controlled-phase error
``delta_phi/pi`` from -0.3 to 0.3.  Separate raw inputs are included for
0.1 dB and 0.2 dB interface loss, both at 0.25 dB per-arm channel loss.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from manuscript_reproduction.common import ordered_columns, parser, read_csv, write_csv


def main() -> None:
    args = parser(__doc__).parse_args()
    frames = []
    for label in ["interface_0p1db", "interface_0p2db"]:
        frame = read_csv(args.data_root, f"phase_error/{label}/raw_label_phase_error_summary.csv")
        frame["source_folder"] = label
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["subfigure"] = "Fig. 3(c,d)"
    data["interface_loss_db_per_reflection"] = data["source_loss_db_per_interface"].astype(float)
    data["channel_loss_db_per_arm_code"] = data["channel_loss_db_per_arm"].astype(float)
    data["end_to_end_channel_loss_db_midpoint"] = 2.0 * data["channel_loss_db_per_arm_code"]
    data = ordered_columns(
        data.sort_values(["interface_loss_db_per_reflection", "M", "phase_error_pi"]).reset_index(drop=True),
        [
            "subfigure",
            "M",
            "constellation",
            "receiver",
            "interface_loss_db_per_reflection",
            "channel_loss_db_per_arm_code",
            "end_to_end_channel_loss_db_midpoint",
            "phase_error_pi",
            "phase_error_rad",
            "optimized_spacing_d",
            "hashing_bound_bits_per_attempt",
            "source_folder",
        ],
    )
    path = write_csv(data, args.outdir, "fig3cd_phase_bias_raw_label_srm.csv")
    print(path)


if __name__ == "__main__":
    main()

