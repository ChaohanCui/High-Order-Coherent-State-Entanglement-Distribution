#!/usr/bin/env python3
"""Create the QAM SRM data table for manuscript Fig. 1(c).

Fig. 1(c) plots achievable hashing bound versus channel loss.  This script
extracts the full-state/raw-label SRM curves, meaning the joint vacuum
component is kept when Charlie's square-root measurement is constructed.

The panel also contains external/theory baselines such as PLOB, CTW,
single-photon, and Hex-GKP curves.  Those baselines are not regenerated here;
publish their final numeric values beside this output if they are part of the
final plotting script.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from manuscript_reproduction.common import (
    add_loss_conventions,
    ordered_columns,
    parser,
    read_csv,
    write_csv,
)


def build(data_root: Path) -> pd.DataFrame:
    low_m = read_csv(
        data_root,
        "ideal_channel_srm/raw_label_vs_vacuum_omit_srm_all_data.csv",
    )
    low_m = low_m[low_m["M"].isin([2, 4, 8, 16])].copy()
    low_m["hashing_bound_bits_per_attempt"] = low_m["raw_rate"]
    low_m["optimized_spacing_d"] = low_m["raw_d"]
    low_m["source_dataset"] = "merged_2_4_8_16_raw_label_srm"

    qam32 = read_csv(data_root, "ideal_channel_srm/qam32_raw_label_vs_vacuum_merged.csv")
    qam32 = qam32.copy()
    qam32["hashing_bound_bits_per_attempt"] = qam32["raw_rate"]
    qam32["optimized_spacing_d"] = qam32["raw_d"]
    qam32["source_dataset"] = "merged_32qam_seeded_dense_refined"

    data = pd.concat([low_m, qam32], ignore_index=True, sort=False)
    data = add_loss_conventions(data, "loss_db")
    data["constellation"] = data["M"].astype(int).astype(str) + "-QAM"
    data["receiver"] = "full_state_raw_label_srm"
    data["subfigure"] = "Fig. 1(c)"

    return ordered_columns(
        data.sort_values(["M", "loss_db_per_arm_code"]).reset_index(drop=True),
        [
            "subfigure",
            "M",
            "constellation",
            "receiver",
            "loss_db_per_arm_code",
            "end_to_end_loss_db_midpoint",
            "hashing_bound_bits_per_attempt",
            "optimized_spacing_d",
            "source_dataset",
        ],
    )


def main() -> None:
    args = parser(__doc__).parse_args()
    path = write_csv(build(args.data_root), args.outdir, "fig1c_qam_srm_hashing_bound.csv")
    print(path)


if __name__ == "__main__":
    main()

