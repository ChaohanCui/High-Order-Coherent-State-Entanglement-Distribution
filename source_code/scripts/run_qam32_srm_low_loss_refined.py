#!/usr/bin/env python3
"""Refine the 32-QAM low-loss SRM comparison to remove sparse-candidate kinks."""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(ROOT / "scripts"))
from run_qam32_srm_dense_extra_points import evaluate_both_receivers


OUTDIR = ROOT / "results" / "raw_label_vs_vacuum_omit_srm_qam32_low_loss_refined"
OUTCSV = OUTDIR / "qam32_raw_label_vs_vacuum_low_loss_refined.csv"


FIELDS = [
    "M",
    "loss_db",
    "vacuum_rate",
    "raw_rate",
    "raw_minus_vacuum",
    "vacuum_d",
    "raw_d",
    "vacuum_candidate_source",
    "raw_candidate_source",
    "vacuum_candidate_seconds",
    "raw_candidate_seconds",
    "note",
]


def target_grid() -> dict[float, list[float]]:
    return {
        0.005: [1.88, 1.92, 1.96, 2.00, 2.04, 2.08],
        0.010: [1.84, 1.88, 1.92, 1.96, 2.00, 2.04],
        0.020: [1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94],
        0.030: [1.78, 1.80, 1.82, 1.84, 1.86, 1.88],
        0.040: [1.76, 1.78, 1.80, 1.82, 1.84, 1.86],
        0.050: [1.74, 1.76, 1.78, 1.80, 1.82, 1.84],
        0.060: [1.72, 1.74, 1.76, 1.78, 1.80, 1.82],
        0.070: [1.70, 1.72, 1.74, 1.76, 1.78, 1.80],
        0.080: [1.68, 1.70, 1.72, 1.74, 1.76, 1.78],
        0.090: [1.66, 1.68, 1.70, 1.72, 1.74, 1.76],
        0.100: [1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76],
    }


def read_completed() -> pd.DataFrame:
    if not OUTCSV.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(OUTCSV)


def write_rows(rows: list[dict[str, object]]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with OUTCSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: float(item["loss_db"])):
            writer.writerow(row)


def evaluate_point(loss_db: float, spacing: float):
    result = evaluate_both_receivers(loss_db, spacing)
    return spacing, result


def row_for_loss(loss_db: float, spacings: list[float]) -> dict[str, object]:
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(evaluate_point, loss_db, spacing) for spacing in spacings]
        for future in as_completed(futures):
            results.append(future.result())

    vacuum_d, vacuum_result = max(results, key=lambda item: item[1].vacuum_rate)
    raw_d, raw_result = max(results, key=lambda item: item[1].raw_rate)
    return {
        "M": 32,
        "loss_db": f"{loss_db:.12g}",
        "vacuum_rate": f"{vacuum_result.vacuum_rate:.12g}",
        "raw_rate": f"{raw_result.raw_rate:.12g}",
        "raw_minus_vacuum": f"{raw_result.raw_rate - vacuum_result.vacuum_rate:.12g}",
        "vacuum_d": f"{vacuum_d:.12g}",
        "raw_d": f"{raw_d:.12g}",
        "vacuum_candidate_source": "low_loss_refined_grid",
        "raw_candidate_source": "low_loss_refined_grid",
        "vacuum_candidate_seconds": f"{vacuum_result.seconds:.3f}",
        "raw_candidate_seconds": f"{raw_result.seconds:.3f}",
        "note": "32-QAM low-loss refined local d grid",
    }


def main() -> None:
    completed = read_completed()
    rows = completed.to_dict("records")
    done = {round(float(loss), 12) for loss in completed.get("loss_db", [])}
    for loss_db, spacings in target_grid().items():
        key = round(loss_db, 12)
        if key in done:
            print(f"skip loss={loss_db:g}", flush=True)
            continue
        row = row_for_loss(loss_db, spacings)
        rows.append(row)
        write_rows(rows)
        print(
            f"loss={loss_db:g}: vacuum={float(row['vacuum_rate']):.9g} "
            f"d={float(row['vacuum_d']):.4g}; raw={float(row['raw_rate']):.9g} "
            f"d={float(row['raw_d']):.4g}; diff={float(row['raw_minus_vacuum']):.9g}",
            flush=True,
        )
    write_rows(rows)
    print(OUTCSV)


if __name__ == "__main__":
    main()
