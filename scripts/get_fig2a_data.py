#!/usr/bin/env python3
"""Create the optimized-spacing table for manuscript Fig. 2(a).

The data source is the same full-state/raw-label SRM optimization as Fig. 1(c),
but this script exports the optimized nearest-neighbor QAM spacing ``d`` rather
than the hashing bound.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from get_fig1c_data import build as build_fig1c

from manuscript_reproduction.common import ordered_columns, parser, write_csv


def main() -> None:
    args = parser(__doc__).parse_args()
    data = build_fig1c(args.data_root)
    data["subfigure"] = "Fig. 2(a)"
    data = ordered_columns(
        data,
        [
            "subfigure",
            "M",
            "constellation",
            "receiver",
            "loss_db_per_arm_code",
            "end_to_end_loss_db_midpoint",
            "optimized_spacing_d",
            "hashing_bound_bits_per_attempt",
            "source_dataset",
        ],
    )
    path = write_csv(data, args.outdir, "fig2a_optimized_spacing_vs_loss.csv")
    print(path)


if __name__ == "__main__":
    main()

