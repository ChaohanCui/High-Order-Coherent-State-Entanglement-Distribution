#!/usr/bin/env python3
"""Create End Matter Fig. 5(a) data: vacuum-omit versus full-state SRM.

The table exports both rates and their difference.  Positive
``raw_minus_vacuum`` means the full-state/raw-label SRM, which keeps the joint
vacuum component, performs better than the vacuum-omitting SRM.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from manuscript_reproduction.common import add_loss_conventions, ordered_columns, parser, read_csv, write_csv


def main() -> None:
    args = parser(__doc__).parse_args()
    low_m = read_csv(args.data_root, "ideal_channel_srm/raw_label_vs_vacuum_omit_srm_all_data.csv")
    qam32 = read_csv(args.data_root, "ideal_channel_srm/qam32_raw_label_vs_vacuum_merged.csv")
    data = pd.concat([low_m, qam32], ignore_index=True, sort=False)
    data = add_loss_conventions(data, "loss_db")
    data["subfigure"] = "End Matter Fig. 5(a)"
    data["constellation"] = data["M"].astype(int).astype(str) + "-QAM"
    data["comparison"] = "full_state_raw_label_srm_minus_vacuum_omit_srm"
    data = ordered_columns(
        data.sort_values(["M", "loss_db_per_arm_code"]).reset_index(drop=True),
        [
            "subfigure",
            "M",
            "constellation",
            "comparison",
            "loss_db_per_arm_code",
            "end_to_end_loss_db_midpoint",
            "raw_rate",
            "vacuum_rate",
            "raw_minus_vacuum",
            "raw_d",
            "vacuum_d",
        ],
    )
    path = write_csv(data, args.outdir, "fig5a_raw_label_vs_vacuum_omit_srm.csv")
    print(path)


if __name__ == "__main__":
    main()

