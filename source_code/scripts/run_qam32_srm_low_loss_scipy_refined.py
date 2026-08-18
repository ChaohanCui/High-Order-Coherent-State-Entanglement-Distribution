#!/usr/bin/env python3
"""Bounded SciPy refinement for 32-QAM low-loss SRM optima."""

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

import pandas as pd
import scipy.optimize


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
from run_qam32_srm_dense_extra_points import CandidateResult, evaluate_both_receivers


SEED_SOURCE = (
    ROOT
    / "results"
    / "raw_label_vs_vacuum_omit_srm_qam32_low_loss_refined"
    / "qam32_raw_label_vs_vacuum_low_loss_refined.csv"
)
OUTDIR = ROOT / "results" / "raw_label_vs_vacuum_omit_srm_qam32_low_loss_scipy_refined"
OUTCSV = OUTDIR / "qam32_raw_label_vs_vacuum_low_loss_scipy_refined.csv"

FIELDS = [
    "M",
    "loss_db",
    "vacuum_rate",
    "raw_rate",
    "raw_minus_vacuum",
    "vacuum_d",
    "raw_d",
    "vacuum_bracket_min",
    "vacuum_bracket_max",
    "raw_bracket_min",
    "raw_bracket_max",
    "vacuum_optimizer_success",
    "raw_optimizer_success",
    "vacuum_optimizer_nfev",
    "raw_optimizer_nfev",
    "cache_evaluations",
    "vacuum_boundary_peak",
    "raw_boundary_peak",
    "seconds",
    "note",
]


@dataclass(frozen=True)
class LossSeed:
    loss_db: float
    vacuum_seed_d: float
    raw_seed_d: float


@dataclass(frozen=True)
class RefinedLoss:
    loss_db: float
    vacuum: CandidateResult
    raw: CandidateResult
    vacuum_bracket: tuple[float, float]
    raw_bracket: tuple[float, float]
    vacuum_success: bool
    raw_success: bool
    vacuum_nfev: int
    raw_nfev: int
    cache_evaluations: int
    vacuum_boundary_peak: bool
    raw_boundary_peak: bool
    seconds: float


class SharedObjective:
    def __init__(self, loss_db: float):
        self.loss_db = loss_db
        self.cache: dict[float, CandidateResult] = {}

    @staticmethod
    def key(spacing: float) -> float:
        return round(float(spacing), 12)

    def evaluate(self, spacing: float) -> CandidateResult:
        key = self.key(spacing)
        if key not in self.cache:
            self.cache[key] = evaluate_both_receivers(self.loss_db, float(spacing))
        return self.cache[key]

    def rate(self, spacing: float, receiver: str) -> float:
        result = self.evaluate(spacing)
        if receiver == "vacuum":
            return result.vacuum_rate
        if receiver == "raw":
            return result.raw_rate
        raise ValueError(f"Unknown receiver {receiver!r}")

    def best(self, receiver: str) -> CandidateResult:
        return max(self.cache.values(), key=lambda result: self.rate(result.d, receiver))


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def low_loss_seeds(path: Path, include_0p1: bool) -> list[LossSeed]:
    df = pd.read_csv(path)
    df["loss_db"] = df["loss_db"].astype(float)
    hi = 0.1000000001 if include_0p1 else 0.1
    df = df[(df["loss_db"] > 0.0) & (df["loss_db"] <= hi)].copy()
    if not include_0p1:
        df = df[df["loss_db"] < 0.1]
    out = []
    for _, row in df.sort_values("loss_db").iterrows():
        out.append(
            LossSeed(
                loss_db=float(row["loss_db"]),
                vacuum_seed_d=float(row["vacuum_d"]),
                raw_seed_d=float(row["raw_d"]),
            )
        )
    return out


def half_width(loss_db: float) -> float:
    if loss_db <= 0.0050001:
        return 0.24
    if loss_db <= 0.0100001:
        return 0.20
    if loss_db <= 0.0300001:
        return 0.14
    return 0.10


def bracket(seed: float, loss_db: float, spacing_min: float, spacing_max: float) -> tuple[float, float]:
    width = half_width(loss_db)
    lo = max(spacing_min, seed - width)
    hi = min(spacing_max, seed + width)
    if not lo < seed < hi:
        lo = max(spacing_min, min(seed, hi) - width)
        hi = min(spacing_max, max(seed, lo) + width)
    if hi <= lo:
        raise ValueError(f"Bad bracket for loss={loss_db:g}, seed={seed:g}: [{lo:g}, {hi:g}]")
    return (lo, hi)


def optimize_receiver(
    objective: SharedObjective,
    receiver: str,
    seed: float,
    bounds: tuple[float, float],
    xatol: float,
    maxiter: int,
) -> tuple[scipy.optimize.OptimizeResult, CandidateResult, bool]:
    lo, hi = bounds
    for spacing in (lo, seed, hi):
        objective.evaluate(spacing)

    opt = scipy.optimize.minimize_scalar(
        lambda d: -objective.rate(float(d), receiver),
        bounds=bounds,
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    if math.isfinite(float(opt.x)):
        objective.evaluate(float(opt.x))

    best = objective.best(receiver)
    edge_tol = max(5.0 * xatol, 1.0e-7)
    boundary = math.isclose(best.d, lo, abs_tol=edge_tol) or math.isclose(
        best.d, hi, abs_tol=edge_tol
    )
    return opt, best, boundary


def refine_loss(
    seed: LossSeed,
    spacing_min: float,
    spacing_max: float,
    xatol: float,
    maxiter: int,
) -> RefinedLoss:
    start = time.perf_counter()
    objective = SharedObjective(seed.loss_db)
    vacuum_bracket = bracket(seed.vacuum_seed_d, seed.loss_db, spacing_min, spacing_max)
    raw_bracket = bracket(seed.raw_seed_d, seed.loss_db, spacing_min, spacing_max)

    vacuum_opt, _vacuum_best, vacuum_boundary = optimize_receiver(
        objective,
        "vacuum",
        seed.vacuum_seed_d,
        vacuum_bracket,
        xatol,
        maxiter,
    )
    raw_opt, _raw_best, raw_boundary = optimize_receiver(
        objective,
        "raw",
        seed.raw_seed_d,
        raw_bracket,
        xatol,
        maxiter,
    )

    # Select after both searches so each receiver can benefit from the other's
    # cached probes when the two optima are close.
    vacuum_best = objective.best("vacuum")
    raw_best = objective.best("raw")
    return RefinedLoss(
        loss_db=seed.loss_db,
        vacuum=vacuum_best,
        raw=raw_best,
        vacuum_bracket=vacuum_bracket,
        raw_bracket=raw_bracket,
        vacuum_success=bool(vacuum_opt.success),
        raw_success=bool(raw_opt.success),
        vacuum_nfev=int(vacuum_opt.nfev),
        raw_nfev=int(raw_opt.nfev),
        cache_evaluations=len(objective.cache),
        vacuum_boundary_peak=vacuum_boundary,
        raw_boundary_peak=raw_boundary,
        seconds=time.perf_counter() - start,
    )


def row_dict(row: RefinedLoss) -> dict[str, object]:
    return {
        "M": 32,
        "loss_db": fmt(row.loss_db),
        "vacuum_rate": fmt(row.vacuum.vacuum_rate),
        "raw_rate": fmt(row.raw.raw_rate),
        "raw_minus_vacuum": fmt(row.raw.raw_rate - row.vacuum.vacuum_rate),
        "vacuum_d": fmt(row.vacuum.d),
        "raw_d": fmt(row.raw.d),
        "vacuum_bracket_min": fmt(row.vacuum_bracket[0]),
        "vacuum_bracket_max": fmt(row.vacuum_bracket[1]),
        "raw_bracket_min": fmt(row.raw_bracket[0]),
        "raw_bracket_max": fmt(row.raw_bracket[1]),
        "vacuum_optimizer_success": int(row.vacuum_success),
        "raw_optimizer_success": int(row.raw_success),
        "vacuum_optimizer_nfev": row.vacuum_nfev,
        "raw_optimizer_nfev": row.raw_nfev,
        "cache_evaluations": row.cache_evaluations,
        "vacuum_boundary_peak": int(row.vacuum_boundary_peak),
        "raw_boundary_peak": int(row.raw_boundary_peak),
        "seconds": f"{row.seconds:.3f}",
        "note": "bounded scipy minimize_scalar around low-loss grid optimum",
    }


def read_completed(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: float(item["loss_db"])):
            writer.writerow({name: row[name] for name in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-source", type=Path, default=SEED_SOURCE)
    parser.add_argument("--outcsv", type=Path, default=OUTCSV)
    parser.add_argument("--spacing-min", type=float, default=1.25)
    parser.add_argument("--spacing-max", type=float, default=2.35)
    parser.add_argument("--xatol", type=float, default=0.003)
    parser.add_argument("--maxiter", type=int, default=14)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--include-0p1", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None

    seeds = low_loss_seeds(args.seed_source, include_0p1=args.include_0p1)
    completed_rows = [] if args.force else read_completed(args.outcsv)
    completed_losses = {round(float(row["loss_db"]), 12) for row in completed_rows}
    pending = [seed for seed in seeds if round(seed.loss_db, 12) not in completed_losses]
    print(f"Total losses: {len(seeds)}; completed: {len(completed_rows)}; pending: {len(pending)}")

    rows: list[dict[str, object]] = list(completed_rows)
    limiter = threadpool_limits(limits=1) if threadpool_limits is not None else None
    if limiter is not None:
        limiter.__enter__()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    refine_loss,
                    seed,
                    args.spacing_min,
                    args.spacing_max,
                    args.xatol,
                    args.maxiter,
                ): seed
                for seed in pending
            }
            for future in as_completed(futures):
                row = future.result()
                rows.append(row_dict(row))
                write_rows(rows, args.outcsv)
                print(
                    f"loss={row.loss_db:g}: vacuum R={row.vacuum.vacuum_rate:.9g} "
                    f"d={row.vacuum.d:.6g}; raw R={row.raw.raw_rate:.9g} "
                    f"d={row.raw.d:.6g}; evals={row.cache_evaluations}; "
                    f"seconds={row.seconds:.1f}",
                    flush=True,
                )
    finally:
        if limiter is not None:
            limiter.__exit__(None, None, None)

    write_rows(rows, args.outcsv)
    print(args.outcsv)


if __name__ == "__main__":
    main()
