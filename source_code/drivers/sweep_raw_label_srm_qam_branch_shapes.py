#!/usr/bin/env python3
"""Raw-label SRM d-sweeps for visualizing QAM branch switches."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT / "scripts"))
from check_raw_label_vs_vacuum_omit_srm_qam import evaluate_receiver


OUTDIR = ROOT / "results" / "raw_label_srm_qam_branch_sweeps"
OUTCSV = OUTDIR / "raw_label_srm_qam_branch_sweep_points.csv"
SUMMARY_CSV = OUTDIR / "raw_label_srm_qam_branch_sweep_local_maxima.csv"
PLOT_PNG = OUTDIR / "raw_label_srm_qam_branch_sweeps_0p9_0p95db.png"

FIELDS = [
    "M",
    "constellation",
    "loss_db",
    "eta",
    "spacing_d",
    "hashing_bound_bits_per_attempt",
    "success_probability",
    "average_target_fidelity",
    "useful_outcomes",
    "grid_region",
    "seconds",
]


@dataclass(frozen=True)
class SweepTask:
    m: int
    loss_db: float
    spacing: float
    region: str


@dataclass(frozen=True)
class SweepResult:
    task: SweepTask
    rate: float
    success_probability: float
    average_fidelity: float
    useful_outcomes: int
    seconds: float


def inclusive_grid(lo: float, hi: float, step: float) -> list[float]:
    count = int(round((hi - lo) / step))
    return [
        round(lo + idx * step, 10)
        for idx in range(count + 1)
        if lo + idx * step <= hi + 1.0e-12
    ]


def region_specs(m: int) -> list[tuple[str, float, float, float]]:
    """Return (label, d_min, d_max, d_step) sweep regions for each QAM size."""
    if m == 4:
        return [
            ("full_d", 0.50, 2.00, 0.0025),
        ]
    if m == 8:
        return [
            ("full_d", 0.40, 2.10, 0.005),
        ]
    if m == 16:
        return [
            ("full_d", 0.30, 2.15, 0.010),
        ]
    raise ValueError(f"Unsupported M={m}")


def make_tasks(ms: list[int], losses: list[float]) -> list[SweepTask]:
    tasks: list[SweepTask] = []
    for m in ms:
        points: dict[float, str] = {}
        for region, lo, hi, step in region_specs(m):
            for spacing in inclusive_grid(lo, hi, step):
                points.setdefault(spacing, region)
        for loss in losses:
            for spacing, region in sorted(points.items()):
                tasks.append(SweepTask(m=m, loss_db=loss, spacing=spacing, region=region))
    return tasks


def task_key(task: SweepTask) -> tuple[int, float, float]:
    return (task.m, round(task.loss_db, 10), round(task.spacing, 10))


def read_completed(path: Path) -> dict[tuple[int, float, float], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[tuple[int, float, float], dict[str, str]] = {}
    for row in rows:
        key = (
            int(row["M"]),
            round(float(row["loss_db"]), 10),
            round(float(row["spacing_d"]), 10),
        )
        out[key] = row
    return out


def evaluate_task(task: SweepTask) -> SweepResult:
    start = time.perf_counter()
    rate, success, fidelity, useful = evaluate_receiver(
        task.m,
        task.loss_db,
        task.spacing,
        "raw_label_srm",
    )
    return SweepResult(
        task=task,
        rate=rate,
        success_probability=success,
        average_fidelity=fidelity,
        useful_outcomes=useful,
        seconds=time.perf_counter() - start,
    )


def row_from_result(result: SweepResult) -> dict[str, object]:
    eta = 10.0 ** (-result.task.loss_db / 10.0)
    return {
        "M": result.task.m,
        "constellation": f"{result.task.m}-QAM",
        "loss_db": f"{result.task.loss_db:.12g}",
        "eta": f"{eta:.12g}",
        "spacing_d": f"{result.task.spacing:.12g}",
        "hashing_bound_bits_per_attempt": f"{result.rate:.12g}",
        "success_probability": f"{result.success_probability:.12g}",
        "average_target_fidelity": f"{result.average_fidelity:.12g}",
        "useful_outcomes": result.useful_outcomes,
        "grid_region": result.task.region,
        "seconds": f"{result.seconds:.3f}",
    }


def write_points(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (int(item["M"]), float(item["loss_db"]), float(item["spacing_d"])),
        ):
            writer.writerow({field: row[field] for field in FIELDS})


def load_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(["M", "loss_db", "spacing_d"]).reset_index(drop=True)


def mark_local_maxima(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, loss), sub in df.groupby(["M", "loss_db"], sort=True):
        sub = sub.sort_values("spacing_d").reset_index(drop=True)
        y = sub["hashing_bound_bits_per_attempt"].astype(float).to_numpy()
        if len(sub) < 3:
            continue
        for idx in range(1, len(sub) - 1):
            if y[idx] >= y[idx - 1] and y[idx] >= y[idx + 1]:
                row = sub.iloc[idx].to_dict()
                row["is_global_maximum"] = 0
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    maxima = pd.DataFrame(rows)
    global_keys = set()
    for (m, loss), sub in maxima.groupby(["M", "loss_db"], sort=True):
        idx = sub["hashing_bound_bits_per_attempt"].astype(float).idxmax()
        global_keys.add(idx)
    maxima["is_global_maximum"] = [int(idx in global_keys) for idx in maxima.index]
    return maxima.sort_values(["M", "loss_db", "hashing_bound_bits_per_attempt"], ascending=[True, True, False])


def write_summary(maxima: pd.DataFrame, path: Path) -> None:
    if maxima.empty:
        path.write_text("")
        return
    cols = [
        "M",
        "constellation",
        "loss_db",
        "spacing_d",
        "hashing_bound_bits_per_attempt",
        "grid_region",
        "success_probability",
        "average_target_fidelity",
        "useful_outcomes",
        "is_global_maximum",
    ]
    maxima[cols].to_csv(path, index=False)


def plot_sweeps(df: pd.DataFrame, maxima: pd.DataFrame, path: Path) -> None:
    ms = [4, 8, 16]
    losses = [0.90, 0.95]
    colors = {0.90: "#1B6CA8", 0.95: "#B23A48"}

    fig, axes = plt.subplots(
        len(ms),
        1,
        figsize=(8.6, 9.2),
        sharex=False,
        constrained_layout=True,
    )
    for ax, m in zip(axes, ms, strict=True):
        for loss in losses:
            sub = df[
                (df["M"].astype(int) == m)
                & np.isclose(df["loss_db"].astype(float), loss)
            ].sort_values("spacing_d")
            if sub.empty:
                continue
            ax.plot(
                sub["spacing_d"],
                sub["hashing_bound_bits_per_attempt"],
                color=colors[loss],
                linewidth=1.8,
                alpha=0.95,
                label=f"{loss:g} dB",
            )

            if not maxima.empty:
                peak = maxima[
                    (maxima["M"].astype(int) == m)
                    & np.isclose(maxima["loss_db"].astype(float), loss)
                ]
                for _, row in peak.iterrows():
                    is_global = bool(int(row["is_global_maximum"]))
                    ax.scatter(
                        [float(row["spacing_d"])],
                        [float(row["hashing_bound_bits_per_attempt"])],
                        color=colors[loss],
                        edgecolor="white" if is_global else None,
                        linewidth=0.9 if is_global else 0.4,
                        marker="o" if is_global else "x",
                        s=62 if is_global else 48,
                        zorder=5,
                    )
                    if is_global:
                        ax.annotate(
                            f"d={float(row['spacing_d']):.3g}",
                            (float(row["spacing_d"]), float(row["hashing_bound_bits_per_attempt"])),
                            textcoords="offset points",
                            xytext=(5, 5),
                            fontsize=8,
                            color=colors[loss],
                        )

        ax.set_title(f"{m}-QAM raw-label SRM d sweep")
        ax.set_xlabel("QAM nearest-neighbor spacing d")
        ax.set_ylabel("Hashing bound (bits/attempt)")
        ax.grid(True, alpha=0.25)
        ax.legend(title="Channel loss", fontsize=8.5)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def parse_list(text: str, cast) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--ms", default="4,8,16")
    parser.add_argument("--losses-db", default="0.9,0.95")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outcsv = args.outdir / OUTCSV.name
    summary_csv = args.outdir / SUMMARY_CSV.name
    plot_png = args.outdir / PLOT_PNG.name

    ms = parse_list(args.ms, int)
    losses = parse_list(args.losses_db, float)
    tasks = make_tasks(ms, losses)

    completed = {} if args.force else read_completed(outcsv)
    rows: list[dict[str, object]] = list(completed.values())
    pending = [task for task in tasks if task_key(task) not in completed]
    print(f"Total tasks: {len(tasks)}; completed: {len(completed)}; pending: {len(pending)}")

    limiter = None
    try:
        try:
            from threadpoolctl import threadpool_limits

            limiter = threadpool_limits(limits=1)
            limiter.__enter__()
        except Exception:
            limiter = None

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(evaluate_task, task): task for task in pending}
            done = 0
            for future in as_completed(futures):
                result = future.result()
                done += 1
                rows.append(row_from_result(result))
                if done == 1 or done % 25 == 0:
                    write_points(rows, outcsv)
                    print(
                        f"{done}/{len(pending)} "
                        f"M={result.task.m} loss={result.task.loss_db:g} "
                        f"d={result.task.spacing:g} R={result.rate:.8g}",
                        flush=True,
                    )
    finally:
        if limiter is not None:
            limiter.__exit__(None, None, None)

    write_points(rows, outcsv)
    df = load_points(outcsv)
    maxima = mark_local_maxima(df)
    write_summary(maxima, summary_csv)
    plot_sweeps(df, maxima, plot_png)
    print(outcsv)
    print(summary_csv)
    print(plot_png)


if __name__ == "__main__":
    main()
