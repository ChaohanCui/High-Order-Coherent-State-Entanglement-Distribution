#!/usr/bin/env python3
"""Raw-label SRM sweep for reflection-source/interface loss.

This mirrors the May-29 reflection-source/interface-loss study, but Charlie's
measurement keeps the vacuum component when constructing the square-root
measurement.  The old vacuum-omit data are left untouched for comparison.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
import scipy.optimize

from qam_reflection_source_loss_hashing import DISPLAY_NAMES, EvalPoint, evaluate_reflection_srm


LOCAL_FIELDS = [
    "M",
    "constellation",
    "receiver",
    "generation_loss_db_per_step",
    "channel_loss_db",
    "branch_label",
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
]


@dataclass(frozen=True)
class BranchJob:
    m: int
    loss_db: float
    channel_loss_db: float
    branch_label: str
    bracket_min: float
    bracket_max: float


@dataclass(frozen=True)
class BranchResult:
    job: BranchJob
    spacing: float
    point: EvalPoint
    optimizer_success: bool
    optimizer_nfev: int
    is_boundary_peak: bool
    is_global_maximum: bool
    seconds: float


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def inclusive_grid(lo: float, hi: float, step: float) -> list[float]:
    count = int(round((hi - lo) / step))
    return [
        round(lo + idx * step, 10)
        for idx in range(count + 1)
        if lo + idx * step <= hi + 1.0e-9
    ]


def build_jobs(channel_loss_db: float, dense: bool = True) -> list[BranchJob]:
    base = inclusive_grid(0.0, 0.5, 0.05)
    losses_by_m: dict[int, list[float]] = {
        2: base,
        4: base,
        8: base,
        16: base,
    }
    if dense:
        losses_by_m[4] = sorted(set(losses_by_m[4] + inclusive_grid(0.28, 0.36, 0.005)))
        losses_by_m[8] = sorted(set(losses_by_m[8] + inclusive_grid(0.18, 0.26, 0.005)))
        losses_by_m[16] = sorted(set(losses_by_m[16] + inclusive_grid(0.04, 0.16, 0.01)))

    branch_specs = {
        2: [("single", 0.80, 2.30)],
        4: [("high_d", 1.10, 2.15), ("low_d", 0.50, 1.30)],
        8: [("high_d", 1.15, 2.15), ("low_d", 0.28, 0.90)],
        16: [("high_d", 1.05, 2.15), ("mid_d", 0.65, 1.30), ("low_d", 0.22, 0.80)],
    }

    jobs: list[BranchJob] = []
    for m, losses in losses_by_m.items():
        for loss_db in losses:
            for branch_label, bracket_min, bracket_max in branch_specs[m]:
                jobs.append(
                    BranchJob(
                        m=m,
                        loss_db=loss_db,
                        channel_loss_db=channel_loss_db,
                        branch_label=branch_label,
                        bracket_min=bracket_min,
                        bracket_max=bracket_max,
                    )
                )
    return jobs


class ObjectiveCache:
    def __init__(self, job: BranchJob, rank_tol: float):
        self.job = job
        self.rank_tol = rank_tol
        self.eta_source = 10.0 ** (-job.loss_db / 10.0)
        self.eta_channel = 10.0 ** (-job.channel_loss_db / 10.0)
        self.values: dict[float, EvalPoint] = {}

    @staticmethod
    def key(spacing: float) -> float:
        return round(float(spacing), 12)

    def evaluate(self, spacing: float) -> EvalPoint:
        key = self.key(spacing)
        if key not in self.values:
            self.values[key] = evaluate_reflection_srm(
                m=self.job.m,
                spacing=float(spacing),
                eta_source=self.eta_source,
                eta_channel=self.eta_channel,
                source_loss_db=self.job.loss_db,
                channel_loss_db=self.job.channel_loss_db,
                convention="reflection",
                rank_tol=self.rank_tol,
                receiver="raw_label_srm",
            )
        return self.values[key]

    def rate(self, spacing: float) -> float:
        return self.evaluate(spacing).result.rate


def refine_branch(job: BranchJob, rank_tol: float, xatol: float, maxiter: int) -> BranchResult:
    start = time.perf_counter()
    cache = ObjectiveCache(job, rank_tol=rank_tol)
    opt = scipy.optimize.minimize_scalar(
        lambda d: -cache.rate(float(d)),
        bounds=(job.bracket_min, job.bracket_max),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    candidates = [job.bracket_min, 0.5 * (job.bracket_min + job.bracket_max), job.bracket_max]
    if math.isfinite(float(opt.x)):
        candidates.append(float(opt.x))
    spacing = max(candidates, key=cache.rate)
    point = cache.evaluate(spacing)
    edge_tol = max(10.0 * xatol, 1.0e-7)
    return BranchResult(
        job=job,
        spacing=spacing,
        point=point,
        optimizer_success=bool(opt.success),
        optimizer_nfev=int(opt.nfev),
        is_boundary_peak=(
            math.isclose(spacing, job.bracket_min, abs_tol=edge_tol)
            or math.isclose(spacing, job.bracket_max, abs_tol=edge_tol)
        ),
        is_global_maximum=False,
        seconds=time.perf_counter() - start,
    )


def mark_global(rows: list[BranchResult]) -> list[BranchResult]:
    best_by_key: dict[tuple[int, float], BranchResult] = {}
    for row in rows:
        key = (row.job.m, row.job.loss_db)
        if key not in best_by_key or row.point.result.rate > best_by_key[key].point.result.rate:
            best_by_key[key] = row
    out: list[BranchResult] = []
    for row in rows:
        best = best_by_key[(row.job.m, row.job.loss_db)]
        out.append(
            BranchResult(
                job=row.job,
                spacing=row.spacing,
                point=row.point,
                optimizer_success=row.optimizer_success,
                optimizer_nfev=row.optimizer_nfev,
                is_boundary_peak=row.is_boundary_peak,
                is_global_maximum=(
                    row.job.branch_label == best.job.branch_label
                    and math.isclose(row.job.loss_db, best.job.loss_db)
                ),
                seconds=row.seconds,
            )
        )
    return out


def row_to_dict(row: BranchResult) -> dict[str, object]:
    result = row.point.result
    return {
        "M": row.job.m,
        "constellation": DISPLAY_NAMES.get(row.job.m, f"{row.job.m}-QAM"),
        "receiver": "raw_label_srm",
        "generation_loss_db_per_step": fmt(row.job.loss_db),
        "channel_loss_db": fmt(row.job.channel_loss_db),
        "branch_label": row.job.branch_label,
        "bracket_min": fmt(row.job.bracket_min),
        "bracket_max": fmt(row.job.bracket_max),
        "spacing_d": fmt(row.spacing),
        "hashing_bound_bits_per_attempt": fmt(result.rate),
        "success_probability": fmt(result.success_probability),
        "average_target_fidelity": fmt(result.average_fidelity),
        "probability_weighted_fidelity": fmt(result.success_probability * result.average_fidelity),
        "min_coherent_information": fmt(result.min_coherent_information),
        "useful_outcomes": result.useful_outcomes,
        "optimizer_success": int(row.optimizer_success),
        "optimizer_nfev": row.optimizer_nfev,
        "is_boundary_peak": int(row.is_boundary_peak),
        "is_global_maximum": int(row.is_global_maximum),
        "seconds": f"{row.seconds:.3f}",
    }


def write_local(rows: list[BranchResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOCAL_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.job.m, r.job.loss_db, r.job.branch_label)):
            writer.writerow(row_to_dict(row))


def write_global(rows: list[BranchResult], path: Path) -> None:
    fieldnames = [
        "M",
        "constellation",
        "receiver",
        "generation_loss_db_per_step",
        "channel_loss_db",
        "branch_label",
        "spacing_d",
        "hashing_bound_bits_per_attempt",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted([r for r in rows if r.is_global_maximum], key=lambda r: (r.job.m, r.job.loss_db)):
            item = row_to_dict(row)
            writer.writerow({name: item[name] for name in fieldnames})


def coerce(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in [
        "M",
        "generation_loss_db_per_step",
        "channel_loss_db",
        "spacing_d",
        "hashing_bound_bits_per_attempt",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column])
    return data


def load_vacuum_reference(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    rename = {}
    if "refined_spacing_d" in data.columns:
        rename["refined_spacing_d"] = "spacing_d"
    if "refined_hashing_bound_bits_per_attempt" in data.columns:
        rename["refined_hashing_bound_bits_per_attempt"] = "hashing_bound_bits_per_attempt"
    data = data.rename(columns=rename)
    keep = [
        "M",
        "constellation",
        "generation_loss_db_per_step",
        "channel_loss_db",
        "spacing_d",
        "hashing_bound_bits_per_attempt",
    ]
    data = data[keep].copy()
    data["receiver"] = "vacuum_omit_srm"
    return coerce(data)


def plot_comparison(raw_global_path: Path, vacuum_path: Path, outdir: Path) -> None:
    raw = coerce(pd.read_csv(raw_global_path))
    vacuum = load_vacuum_reference(vacuum_path)
    colors = {2: "#1B6CA8", 4: "#2F9E44", 8: "#D65A31", 16: "#7B2CBF"}

    for metric, ylabel, filename in [
        (
            "hashing_bound_bits_per_attempt",
            "Optimized hashing bound (bits/attempt)",
            "reflection_source_raw_vs_vacuum_hashing_bound.png",
        ),
        (
            "spacing_d",
            "Optimized spacing d",
            "reflection_source_raw_vs_vacuum_spacing_d.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 4.7), constrained_layout=True)
        for m in [2, 4, 8, 16]:
            raw_sub = raw[raw["M"] == m].sort_values("generation_loss_db_per_step")
            vac_sub = vacuum[vacuum["M"] == m].sort_values("generation_loss_db_per_step")
            if not vac_sub.empty:
                ax.plot(
                    vac_sub["generation_loss_db_per_step"],
                    vac_sub[metric],
                    color=colors[m],
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.55,
                    label=f"{m}-QAM vacuum-omit" if metric == "hashing_bound_bits_per_attempt" else None,
                )
            if not raw_sub.empty:
                ax.plot(
                    raw_sub["generation_loss_db_per_step"],
                    raw_sub[metric],
                    color=colors[m],
                    linestyle="-",
                    marker="o",
                    markersize=3.0,
                    linewidth=2.0,
                    label=f"{m}-QAM raw-label" if metric == "hashing_bound_bits_per_attempt" else f"{m}-QAM",
                )
        ax.set_xlabel("Interface/source-generation loss per reflection (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title("Raw-label SRM vs vacuum-omit SRM for interface loss")
        ax.grid(True, alpha=0.25)
        if metric == "hashing_bound_bits_per_attempt":
            ax.legend(ncol=2, fontsize=8)
        else:
            handles = [
                plt.Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="raw-label SRM"),
                plt.Line2D([0], [0], color="black", lw=1.6, linestyle="--", alpha=0.55, label="vacuum-omit SRM"),
            ]
            handles.extend(
                plt.Line2D([0], [0], color=colors[m], lw=2.0, label=f"{m}-QAM")
                for m in [2, 4, 8, 16]
            )
            ax.legend(handles=handles, ncol=2, fontsize=8)
        fig.savefig(outdir / filename, dpi=220)
        plt.close(fig)

    merged = raw.merge(
        vacuum,
        on=["M", "generation_loss_db_per_step"],
        suffixes=("_raw", "_vacuum"),
        how="inner",
    )
    merged["raw_minus_vacuum"] = (
        merged["hashing_bound_bits_per_attempt_raw"]
        - merged["hashing_bound_bits_per_attempt_vacuum"]
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    for m in [2, 4, 8, 16]:
        sub = merged[merged["M"] == m].sort_values("generation_loss_db_per_step")
        if sub.empty:
            continue
        ax.plot(
            sub["generation_loss_db_per_step"],
            sub["raw_minus_vacuum"],
            color=colors[m],
            marker="o",
            markersize=3.0,
            linewidth=2.0,
            label=f"{m}-QAM",
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("Interface/source-generation loss per reflection (dB)")
    ax.set_ylabel("Optimized hashing bound difference\n(raw-label SRM - vacuum-omit SRM)")
    ax.set_title("Raw-label SRM improvement for interface loss")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(outdir / "reflection_source_raw_minus_vacuum_hashing_bound.png", dpi=220)
    plt.close(fig)
    merged.to_csv(outdir / "reflection_source_raw_vs_vacuum_matched.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/raw_label_reflection_source_loss"))
    parser.add_argument(
        "--vacuum-reference",
        type=Path,
        default=Path("results/refined_reflection_source_loss/reflection_source_refined_dense_transition_global_spacing_ultradense4_16.csv"),
    )
    parser.add_argument("--channel-loss-db", type=float, default=0.25)
    parser.add_argument("--loss-rank-tol", type=float, default=1.0e-8)
    parser.add_argument("--xatol", type=float, default=2.0e-4)
    parser.add_argument("--maxiter", type=int, default=70)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    local_path = args.outdir / "reflection_source_raw_label_local_branches.csv"
    global_path = args.outdir / "reflection_source_raw_label_global_optima.csv"

    if not args.plot_only:
        jobs = build_jobs(args.channel_loss_db, dense=not args.no_dense)
        rows: list[BranchResult] = []

        def run(job: BranchJob) -> BranchResult:
            return refine_branch(job, args.loss_rank_tol, args.xatol, args.maxiter)

        if args.workers and args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(run, job): job for job in jobs}
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    marked = mark_global(rows)
                    write_local(marked, local_path)
                    write_global(marked, global_path)
                    print(
                        f"{DISPLAY_NAMES.get(row.job.m, f'{row.job.m}-QAM')} "
                        f"loss={row.job.loss_db:g} {row.job.branch_label}: "
                        f"d={row.spacing:.6g}, R={row.point.result.rate:.6g}",
                        flush=True,
                    )
        else:
            for job in jobs:
                row = run(job)
                rows.append(row)
                marked = mark_global(rows)
                write_local(marked, local_path)
                write_global(marked, global_path)
                print(
                    f"{DISPLAY_NAMES.get(row.job.m, f'{row.job.m}-QAM')} "
                    f"loss={row.job.loss_db:g} {row.job.branch_label}: "
                    f"d={row.spacing:.6g}, R={row.point.result.rate:.6g}",
                    flush=True,
                )

        rows = mark_global(rows)
        write_local(rows, local_path)
        write_global(rows, global_path)

    plot_comparison(global_path, args.vacuum_reference, args.outdir)
    print(f"Wrote {local_path}")
    print(f"Wrote {global_path}")
    print(f"Wrote {args.outdir / 'reflection_source_raw_vs_vacuum_hashing_bound.png'}")
    print(f"Wrote {args.outdir / 'reflection_source_raw_vs_vacuum_spacing_d.png'}")
    print(f"Wrote {args.outdir / 'reflection_source_raw_minus_vacuum_hashing_bound.png'}")


if __name__ == "__main__":
    main()
