#!/usr/bin/env python3
"""Ultra-dense 16-QAM raw-label SRM interface-loss transition scan."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
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
import pandas as pd

from raw_label_reflection_source_loss_srm import (
    LOCAL_FIELDS,
    BranchJob,
    BranchResult,
    fmt,
    refine_branch,
    row_to_dict,
)


M = 16
COLOR = "#7B2CBF"


def inclusive_grid(lo: float, hi: float, step: float) -> list[float]:
    count = int(round((hi - lo) / step))
    return [
        round(lo + idx * step, 10)
        for idx in range(count + 1)
        if lo + idx * step <= hi + 1.0e-9
    ]


def build_jobs(loss_min: float, loss_max: float, step: float, channel_loss_db: float) -> list[BranchJob]:
    jobs: list[BranchJob] = []
    for loss in inclusive_grid(loss_min, loss_max, step):
        jobs.extend(
            [
                BranchJob(M, loss, channel_loss_db, "high_d", 1.00, 1.85),
                BranchJob(M, loss, channel_loss_db, "mid_d", 0.65, 1.35),
                BranchJob(M, loss, channel_loss_db, "low_d", 0.35, 0.85),
            ]
        )
    return jobs


def coerce_numeric(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in [
        "M",
        "generation_loss_db_per_step",
        "channel_loss_db",
        "bracket_min",
        "bracket_max",
        "spacing_d",
        "hashing_bound_bits_per_attempt",
        "success_probability",
        "average_target_fidelity",
        "probability_weighted_fidelity",
        "min_coherent_information",
        "useful_outcomes",
        "optimizer_success",
        "optimizer_nfev",
        "is_boundary_peak",
        "is_global_maximum",
        "seconds",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column])
    return data


def mark_global_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = coerce_numeric(data)
    if data.empty:
        return data
    data["is_global_maximum"] = 0
    idx = data.groupby(["M", "generation_loss_db_per_step"])[
        "hashing_bound_bits_per_attempt"
    ].idxmax()
    data.loc[idx, "is_global_maximum"] = 1
    return data.sort_values(["M", "generation_loss_db_per_step", "branch_label"]).reset_index(drop=True)


def completed_keys(data: pd.DataFrame) -> set[tuple[float, str]]:
    if data.empty:
        return set()
    data = coerce_numeric(data)
    return {
        (round(float(row.generation_loss_db_per_step), 10), str(row.branch_label))
        for row in data.itertuples()
    }


def write_local_dataframe(data: pd.DataFrame, path: Path) -> None:
    data = mark_global_dataframe(data)
    data.to_csv(path, index=False, columns=LOCAL_FIELDS)


def write_global_dataframe(data: pd.DataFrame, path: Path) -> None:
    data = mark_global_dataframe(data)
    fields = [
        "M",
        "constellation",
        "receiver",
        "generation_loss_db_per_step",
        "channel_loss_db",
        "branch_label",
        "spacing_d",
        "hashing_bound_bits_per_attempt",
    ]
    data[data["is_global_maximum"].astype(int) == 1][fields].to_csv(path, index=False)


def branch_crossing(data: pd.DataFrame, left: str, right: str) -> float | None:
    data = coerce_numeric(data)
    piv = data.pivot_table(
        index="generation_loss_db_per_step",
        columns="branch_label",
        values="hashing_bound_bits_per_attempt",
    )
    if left not in piv or right not in piv:
        return None
    diff = (piv[left] - piv[right]).dropna()
    pairs = list(diff.items())
    for (x1, d1), (x2, d2) in zip(pairs, pairs[1:]):
        if d1 == 0:
            return float(x1)
        if d1 * d2 < 0:
            return float(x1 + (0.0 - d1) * (x2 - x1) / (d2 - d1))
    return None


def load_vacuum_global(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    data = data[data["M"].astype(int) == M].copy()
    return coerce_numeric(data)


def plot_dataframe(data: pd.DataFrame, vacuum_global: pd.DataFrame, out_path: Path) -> None:
    data = mark_global_dataframe(data)
    global_data = data[data["is_global_maximum"].astype(int) == 1].sort_values(
        "generation_loss_db_per_step"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0], "spacing_d", "Optimized spacing d"),
        (axes[1], "hashing_bound_bits_per_attempt", "Hashing bound (bits/attempt)"),
    ]:
        for branch in ["high_d", "mid_d", "low_d"]:
            sub = data[data["branch_label"] == branch].sort_values(
                "generation_loss_db_per_step"
            )
            if sub.empty:
                continue
            ax.plot(
                sub["generation_loss_db_per_step"],
                sub[metric],
                marker="o",
                markersize=2.4,
                linewidth=1.3,
                color=COLOR,
                alpha=0.78 if branch == "high_d" else 0.38,
                label=f"raw {branch}" if metric == "spacing_d" else None,
            )

        pts = list(zip(global_data["generation_loss_db_per_step"], global_data[metric]))
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            jump = abs(y2 - y1) > (0.18 if metric == "spacing_d" else 0.08)
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linewidth=1.1 if jump else 2.4,
                linestyle=":" if jump else "-",
                alpha=0.95,
                label="raw global" if metric == "spacing_d" and (x1, y1) == pts[0] else None,
            )

        if not vacuum_global.empty:
            vac = vacuum_global.sort_values("generation_loss_db_per_step")
            ax.plot(
                vac["generation_loss_db_per_step"],
                vac[metric],
                color="#868e96",
                linestyle="--",
                marker="s",
                markersize=2.5,
                linewidth=1.7,
                alpha=0.85,
                label="vacuum global" if metric == "spacing_d" else None,
            )

        ax.set_xlabel("Interface loss per reflection (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_title("16-QAM raw-label local branches")
    axes[1].set_title("16-QAM raw-label branch rates")
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/raw_label_reflection_source_loss"))
    parser.add_argument(
        "--vacuum-global",
        type=Path,
        default=Path("results/refined_reflection_source_loss/reflection_source_16qam_ultradense_transition_global.csv"),
    )
    parser.add_argument("--loss-min", type=float, default=0.05)
    parser.add_argument("--loss-max", type=float, default=0.15)
    parser.add_argument("--loss-step", type=float, default=0.0025)
    parser.add_argument("--channel-loss-db", type=float, default=0.25)
    parser.add_argument("--loss-rank-tol", type=float, default=1.0e-8)
    parser.add_argument("--xatol", type=float, default=2.0e-4)
    parser.add_argument("--maxiter", type=int, default=70)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    local_path = args.outdir / "reflection_source_16qam_raw_label_ultradense_transition_branches.csv"
    global_path = args.outdir / "reflection_source_16qam_raw_label_ultradense_transition_global.csv"
    plot_path = args.outdir / "reflection_source_16qam_raw_label_ultradense_transition_diagnostic.png"

    if args.plot_only:
        data = pd.read_csv(local_path)
        write_local_dataframe(data, local_path)
        write_global_dataframe(data, global_path)
        plot_dataframe(data, load_vacuum_global(args.vacuum_global), plot_path)
    else:
        if local_path.exists() and not args.restart:
            data = pd.read_csv(local_path)
        else:
            data = pd.DataFrame(columns=LOCAL_FIELDS)
        done = completed_keys(data)
        jobs = [
            job
            for job in build_jobs(args.loss_min, args.loss_max, args.loss_step, args.channel_loss_db)
            if (round(job.loss_db, 10), job.branch_label) not in done
        ]

        def run(job: BranchJob) -> BranchResult:
            return refine_branch(job, args.loss_rank_tol, args.xatol, args.maxiter)

        rows = data.to_dict("records")
        if args.workers and args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(run, job): job for job in jobs}
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row_to_dict(row))
                    current = mark_global_dataframe(pd.DataFrame(rows))
                    write_local_dataframe(current, local_path)
                    write_global_dataframe(current, global_path)
                    print(
                        f"16-QAM raw loss={row.job.loss_db:g} {row.job.branch_label}: "
                        f"d={row.spacing:.6g}, R={row.point.result.rate:.6g}",
                        flush=True,
                    )
        else:
            for job in jobs:
                row = run(job)
                rows.append(row_to_dict(row))
                current = mark_global_dataframe(pd.DataFrame(rows))
                write_local_dataframe(current, local_path)
                write_global_dataframe(current, global_path)
                print(
                    f"16-QAM raw loss={row.job.loss_db:g} {row.job.branch_label}: "
                    f"d={row.spacing:.6g}, R={row.point.result.rate:.6g}",
                    flush=True,
                )

        data = mark_global_dataframe(pd.DataFrame(rows))
        write_local_dataframe(data, local_path)
        write_global_dataframe(data, global_path)
        plot_dataframe(data, load_vacuum_global(args.vacuum_global), plot_path)

    data = pd.read_csv(local_path)
    high_mid = branch_crossing(data, "high_d", "mid_d")
    mid_low = branch_crossing(data, "mid_d", "low_d")
    print(
        f"High/mid crossing estimate: "
        f"{fmt(high_mid) if high_mid is not None else 'not found'} dB"
    )
    print(
        f"Mid/low crossing estimate: "
        f"{fmt(mid_low) if mid_low is not None else 'not found'} dB"
    )
    print(f"Wrote {local_path}")
    print(f"Wrote {global_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
