#!/usr/bin/env python3
"""Scan controlled-phase errors in the reflection-source interface model."""

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

from qam_reflection_source_loss_hashing import (
    DISPLAY_NAMES,
    EvalPoint,
    evaluate_reflection_srm,
)


FIELDS = [
    "M",
    "constellation",
    "receiver",
    "phase_error_pi",
    "phase_error_rad",
    "source_loss_db_per_interface",
    "channel_loss_db_per_arm",
    "eta_source",
    "eta_channel",
    "seed_spacing_d",
    "bracket_min",
    "bracket_max",
    "optimized_spacing_d",
    "hashing_bound_bits_per_attempt",
    "success_probability",
    "average_target_fidelity",
    "probability_weighted_fidelity",
    "min_coherent_information",
    "useful_outcomes",
    "optimizer_success",
    "optimizer_nfev",
    "is_boundary_peak",
    "seconds",
]


@dataclass(frozen=True)
class PhaseJob:
    m: int
    phase_error_pi: float
    seed_spacing: float
    source_loss_db: float
    channel_loss_db: float
    bracket_width: float
    receiver: str


@dataclass(frozen=True)
class PhaseResult:
    job: PhaseJob
    bracket_min: float
    bracket_max: float
    spacing: float
    point: EvalPoint
    optimizer_success: bool
    optimizer_nfev: int
    is_boundary_peak: bool
    seconds: float


class ObjectiveCache:
    def __init__(self, job: PhaseJob, rank_tol: float):
        self.job = job
        self.rank_tol = rank_tol
        self.eta_source = 10.0 ** (-job.source_loss_db / 10.0)
        self.eta_channel = 10.0 ** (-job.channel_loss_db / 10.0)
        self.phase_error_rad = math.pi * job.phase_error_pi
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
                source_loss_db=self.job.source_loss_db,
                channel_loss_db=self.job.channel_loss_db,
                convention="reflection",
                rank_tol=self.rank_tol,
                phase_error_rad=self.phase_error_rad,
                receiver=self.job.receiver,
            )
        return self.values[key]

    def rate(self, spacing: float) -> float:
        return self.evaluate(spacing).result.rate


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def phase_grid(lo: float, hi: float, step: float) -> list[float]:
    count = int(round((hi - lo) / step))
    return [
        round(lo + idx * step, 10)
        for idx in range(count + 1)
        if lo + idx * step <= hi + 1.0e-12
    ]


def load_seed_spacings(path: Path, ms: list[int], source_loss_db: float) -> dict[int, float]:
    df = pd.read_csv(path)
    loss_col = (
        "generation_loss_db_per_step"
        if "generation_loss_db_per_step" in df.columns
        else "source_loss_db"
        if "source_loss_db" in df.columns
        else "source_loss_db_per_interface"
    )
    spacing_col = (
        "refined_spacing_d"
        if "refined_spacing_d" in df.columns
        else "spacing_d"
        if "spacing_d" in df.columns
        else "optimized_spacing_d"
    )
    out: dict[int, float] = {}
    for m in ms:
        sub = df[
            (df["M"].astype(int) == m)
            & (abs(df[loss_col].astype(float) - source_loss_db) < 1.0e-10)
        ]
        if sub.empty:
            raise ValueError(f"No seed spacing for M={m}, source loss={source_loss_db:g} dB")
        out[m] = float(sub.iloc[0][spacing_col])
    return out


def run_job(job: PhaseJob, rank_tol: float, xatol: float, maxiter: int) -> PhaseResult:
    start = time.perf_counter()
    cache = ObjectiveCache(job, rank_tol=rank_tol)
    bracket_min = max(0.02, job.seed_spacing - job.bracket_width)
    bracket_max = job.seed_spacing + job.bracket_width
    opt = scipy.optimize.minimize_scalar(
        lambda d: -cache.rate(float(d)),
        bounds=(bracket_min, bracket_max),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    candidates = [bracket_min, job.seed_spacing, 0.5 * (bracket_min + bracket_max), bracket_max]
    if math.isfinite(float(opt.x)):
        candidates.append(float(opt.x))
    spacing = max(candidates, key=cache.rate)
    point = cache.evaluate(spacing)
    edge_tol = max(10.0 * xatol, 1.0e-7)
    return PhaseResult(
        job=job,
        bracket_min=bracket_min,
        bracket_max=bracket_max,
        spacing=spacing,
        point=point,
        optimizer_success=bool(opt.success),
        optimizer_nfev=int(opt.nfev),
        is_boundary_peak=(
            math.isclose(spacing, bracket_min, abs_tol=edge_tol)
            or math.isclose(spacing, bracket_max, abs_tol=edge_tol)
        ),
        seconds=time.perf_counter() - start,
    )


def row_to_dict(row: PhaseResult) -> dict[str, object]:
    result = row.point.result
    return {
        "M": row.job.m,
        "constellation": DISPLAY_NAMES.get(row.job.m, f"{row.job.m}-QAM"),
        "receiver": row.job.receiver,
        "phase_error_pi": fmt(row.job.phase_error_pi),
        "phase_error_rad": fmt(math.pi * row.job.phase_error_pi),
        "source_loss_db_per_interface": fmt(row.job.source_loss_db),
        "channel_loss_db_per_arm": fmt(row.job.channel_loss_db),
        "eta_source": fmt(row.point.eta_source),
        "eta_channel": fmt(row.point.eta_channel),
        "seed_spacing_d": fmt(row.job.seed_spacing),
        "bracket_min": fmt(row.bracket_min),
        "bracket_max": fmt(row.bracket_max),
        "optimized_spacing_d": fmt(row.spacing),
        "hashing_bound_bits_per_attempt": fmt(result.rate),
        "success_probability": fmt(result.success_probability),
        "average_target_fidelity": fmt(result.average_fidelity),
        "probability_weighted_fidelity": fmt(
            result.success_probability * result.average_fidelity
        ),
        "min_coherent_information": fmt(result.min_coherent_information),
        "useful_outcomes": result.useful_outcomes,
        "optimizer_success": int(row.optimizer_success),
        "optimizer_nfev": row.optimizer_nfev,
        "is_boundary_peak": int(row.is_boundary_peak),
        "seconds": f"{row.seconds:.3f}",
    }


def write_summary(rows: list[PhaseResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.job.m, r.job.phase_error_pi)):
            writer.writerow(row_to_dict(row))


def plot_quantity(df: pd.DataFrame, path: Path, y_col: str, y_label: str) -> None:
    colors = {2: "#1B6CA8", 4: "#2F9E44", 8: "#D65A31"}
    fig, ax = plt.subplots(figsize=(7.6, 4.5), constrained_layout=True)
    for m in sorted(df["M"].unique()):
        sub = df[df["M"] == m].sort_values("phase_error_pi")
        ax.plot(
            sub["phase_error_pi"],
            sub[y_col],
            marker="o",
            markersize=3.4,
            linewidth=2.2,
            color=colors.get(int(m), "#333333"),
            label=f"{int(m)}-QAM",
        )
    ax.axvline(0.0, color="#495057", linewidth=0.9, alpha=0.55)
    ax.set_xlabel(r"Conditional phase error $\delta_\phi / \pi$")
    ax.set_ylabel(y_label)
    ax.set_title("Reflection-source interface phase-error scan")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_mathematica(df: pd.DataFrame, path: Path) -> None:
    qams = [int(m) for m in sorted(df["M"].unique())]
    source_loss_db = float(df["source_loss_db_per_interface"].iloc[0])
    channel_loss_db = float(df["channel_loss_db_per_arm"].iloc[0])

    def format_vector(values: list[float], indent: str = "      ", per_line: int = 8) -> str:
        vals = [fmt(value) for value in values]
        return ",\n".join(
            indent + ", ".join(vals[idx : idx + per_line])
            for idx in range(0, len(vals), per_line)
        )

    def format_data(name: str, column: str, label: str) -> str:
        blocks = []
        for m in qams:
            sub = df[df["M"] == m].sort_values("phase_error_pi")
            blocks.append(
                f"  (* {m}-QAM {label} *)\n"
                "  {\n"
                "    {\n"
                f"{format_vector(sub['phase_error_pi'].tolist())}\n"
                "    },\n"
                "    {\n"
                f"{format_vector(sub[column].tolist())}\n"
                "    }\n"
                "  }"
            )
        return f"{name} = {{\n" + ",\n".join(blocks) + "\n};"

    text = (
        f"(* Reflection-source phase-error scan, source/interface loss = {fmt(source_loss_db)} dB, "
        f"channel/per-arm loss = {fmt(channel_loss_db)} dB. *)\n"
        "(* Each curve is a two-row matrix {{deltaPhi/Pi values}, {curve values}}. *)\n\n"
        f"qams = {{{', '.join(str(m) for m in qams)}}};\n"
        f"sourceLossDb = {fmt(source_loss_db)};\n"
        f"channelLossDb = {fmt(channel_loss_db)};\n\n"
        + format_data("dPhaseErrorData", "optimized_spacing_d", "optimized d")
        + "\n\n"
        + format_data(
            "ratePhaseErrorData",
            "hashing_bound_bits_per_attempt",
            "hashing bound",
        )
        + "\n"
    )
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/reflection_source_phase_error"))
    parser.add_argument(
        "--seed-summary",
        type=Path,
        default=Path("results/refined_reflection_source_loss/reflection_source_refined_summary.csv"),
    )
    parser.add_argument("--ms", default="2,4,8")
    parser.add_argument("--source-loss-db", type=float, default=0.1)
    parser.add_argument("--channel-loss-db", type=float, default=0.25)
    parser.add_argument(
        "--receiver",
        choices=["vacuum_omit_srm", "raw_label_srm"],
        default="vacuum_omit_srm",
    )
    parser.add_argument("--phase-min-pi", type=float, default=-0.05)
    parser.add_argument("--phase-max-pi", type=float, default=0.05)
    parser.add_argument("--phase-step-pi", type=float, default=0.005)
    parser.add_argument("--bracket-width", type=float, default=0.30)
    parser.add_argument("--loss-rank-tol", type=float, default=1.0e-8)
    parser.add_argument("--xatol", type=float, default=1.5e-4)
    parser.add_argument("--maxiter", type=int, default=70)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    ms = [int(item.strip()) for item in args.ms.split(",") if item.strip()]
    seeds = load_seed_spacings(args.seed_summary, ms, args.source_loss_db)
    phases = phase_grid(args.phase_min_pi, args.phase_max_pi, args.phase_step_pi)
    jobs = [
        PhaseJob(
            m=m,
            phase_error_pi=phase,
            seed_spacing=seeds[m],
            source_loss_db=args.source_loss_db,
            channel_loss_db=args.channel_loss_db,
            bracket_width=args.bracket_width,
            receiver=args.receiver,
        )
        for m in ms
        for phase in phases
    ]

    rows: list[PhaseResult] = []
    summary_path = args.outdir / "reflection_source_phase_error_summary.csv"

    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_job, job, args.loss_rank_tol, args.xatol, args.maxiter): job
                for job in jobs
            }
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                write_summary(rows, summary_path)
                print(
                    f"{DISPLAY_NAMES.get(row.job.m, f'{row.job.m}-QAM')} "
                    f"delta/pi={row.job.phase_error_pi:+.4f}: "
                    f"R={row.point.result.rate:.6g}, d={row.spacing:.6g}"
                )
    else:
        for job in jobs:
            row = run_job(job, args.loss_rank_tol, args.xatol, args.maxiter)
            rows.append(row)
            write_summary(rows, summary_path)
            print(
                f"{DISPLAY_NAMES.get(row.job.m, f'{row.job.m}-QAM')} "
                f"delta/pi={row.job.phase_error_pi:+.4f}: "
                f"R={row.point.result.rate:.6g}, d={row.spacing:.6g}"
            )

    write_summary(rows, summary_path)
    df = pd.read_csv(summary_path)
    plot_quantity(
        df,
        args.outdir / "reflection_source_phase_error_hashing_bound.png",
        "hashing_bound_bits_per_attempt",
        "Optimized hashing bound (bits/attempt)",
    )
    plot_quantity(
        df,
        args.outdir / "reflection_source_phase_error_optimized_d.png",
        "optimized_spacing_d",
        "Optimized spacing d",
    )
    write_mathematica(
        df, args.outdir / "mathematica_reflection_source_phase_error_data.wl"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.outdir / 'reflection_source_phase_error_hashing_bound.png'}")
    print(f"Wrote {args.outdir / 'reflection_source_phase_error_optimized_d.png'}")
    print(f"Wrote {args.outdir / 'mathematica_reflection_source_phase_error_data.wl'}")


if __name__ == "__main__":
    main()
