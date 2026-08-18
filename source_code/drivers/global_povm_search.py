#!/usr/bin/env python3
"""Global-search driver for QAM C-only POVM optimization.

This script does not turn the hashing-bound POVM problem into a certified
convex global optimization problem.  Instead, it implements a stronger
reproducible global-search workflow around the nonconvex Stiefel POVM
parametrization:

  1. Sweep one or more outcome counts.
  2. Screen many deterministic and random starts with a modest iteration budget.
  3. Polish the best screened candidates with a larger iteration budget.
  4. Save every start, every polish run, and the best POVM matrix.
  5. Report reproducibility diagnostics for how often the best basin was found.

The result is a high-confidence "best found" lower bound, not a mathematical
certificate of global optimality.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
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
import numpy as np
import pandas as pd

import optimize_qam4_general_povm as povm


SUMMARY_FIELDS = [
    "M",
    "loss_db",
    "eta",
    "spacing_d",
    "reference_spacing_d",
    "spacing_scale",
    "outcomes",
    "previous_ykl_rate",
    "best_rate",
    "best_minus_previous_ykl",
    "relative_gain_vs_previous_percent",
    "best_source",
    "best_start_name",
    "best_polish_name",
    "screen_starts",
    "polished_candidates",
    "screen_best_rate",
    "screen_ykl_rate",
    "polish_ykl_rate",
    "best_minus_screen_ykl",
    "best_minus_polish_ykl",
    "near_best_screen_count",
    "near_best_polish_count",
    "near_best_abs_tol",
    "best_gradient_norm",
    "best_iterations",
    "seconds",
    "best_matrix_path",
]


DETAIL_FIELDS = [
    "M",
    "loss_db",
    "eta",
    "spacing_d",
    "reference_spacing_d",
    "spacing_scale",
    "outcomes",
    "stage",
    "source_name",
    "start_name",
    "rate",
    "rate_delta_vs_previous_ykl",
    "relative_gain_vs_previous_percent",
    "success_probability",
    "useful_outcomes",
    "iterations",
    "gradient_norm",
    "seconds",
]


@dataclass
class SearchCandidate:
    stage: str
    source_name: str
    start_name: str
    a: np.ndarray
    rate: float
    success_probability: float
    useful_outcomes: int
    iterations: int
    gradient_norm: float
    seconds: float


@dataclass(frozen=True)
class SearchSummary:
    m: int
    loss_db: float
    eta: float
    spacing: float
    reference_spacing: float
    spacing_scale: float
    outcomes: int
    previous_ykl_rate: float
    best: SearchCandidate
    screen_best_rate: float
    screen_ykl_rate: float
    polish_ykl_rate: float
    screen_count: int
    polish_count: int
    near_best_screen_count: int
    near_best_polish_count: int
    near_best_abs_tol: float
    seconds: float
    best_matrix_path: Path


def set_m(m: int) -> None:
    povm.M = m
    povm.DIM = m * m


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_outcomes_list(text: str, m: int) -> list[int]:
    dim = m * m
    out: list[int] = []
    for item in text.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key in {"dim", "k"}:
            out.append(dim)
        elif key in {"2dim", "2k", "default"}:
            out.append(2 * dim)
        elif key in {"shor", "davies", "max"}:
            out.append(dim * (dim + 1) // 2)
        else:
            out.append(int(key))
    unique = sorted(set(out))
    for n in unique:
        if n < dim:
            raise ValueError(f"Outcome count {n} is below support dimension {dim}.")
    return unique


def load_previous(path: Path, m: int, losses: list[float]) -> dict[float, tuple[float, float]]:
    df = pd.read_csv(path)
    out: dict[float, tuple[float, float]] = {}
    for loss in losses:
        sub = df[
            (df["M"].astype(int) == m)
            & (np.abs(df["loss_db"].astype(float) - loss) < 1.0e-10)
        ]
        if sub.empty:
            raise ValueError(f"No previous {m}-QAM result for loss={loss:g} dB.")
        row = sub.iloc[0]
        out[round(loss, 10)] = (float(row["spacing_d"]), float(row["rate"]))
    return out


def run_local(
    initial: np.ndarray,
    problem: povm.Problem,
    stage: str,
    source_name: str,
    start_name: str,
    max_iter: int,
    grad_tol: float,
    initial_step: float,
) -> SearchCandidate:
    start = time.perf_counter()
    a, rate, success, useful, iterations, grad_norm = povm.stiefel_ascent(
        initial,
        problem,
        max_iter=max_iter,
        grad_tol=grad_tol,
        initial_step=initial_step,
    )
    return SearchCandidate(
        stage=stage,
        source_name=source_name,
        start_name=start_name,
        a=a,
        rate=rate,
        success_probability=success,
        useful_outcomes=useful,
        iterations=iterations,
        gradient_norm=grad_norm,
        seconds=time.perf_counter() - start,
    )


def generate_screen_starts(
    problem: povm.Problem,
    outcomes: int,
    random_starts: int,
    ykl_perturbations: int,
    perturb_scale: float,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    ykl_initial = povm.split_rows_to_count(
        povm.complete_rows_from_ykl(problem.ykl_rows, problem.ykl_scale),
        outcomes,
    )
    starts: list[tuple[str, np.ndarray]] = [("ykl_complete", ykl_initial)]
    for idx in range(ykl_perturbations):
        starts.append(
            (
                f"ykl_perturb_{idx + 1}",
                povm.perturb_stiefel(ykl_initial, rng, perturb_scale),
            )
        )
    for idx in range(random_starts):
        starts.append(
            (
                f"random_{idx + 1}",
                povm.random_stiefel(outcomes, problem.r.shape[0], rng),
            )
        )
    return starts


def unique_top_candidates(
    candidates: list[SearchCandidate],
    top_k: int,
    rate_tol: float,
) -> list[SearchCandidate]:
    selected: list[SearchCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.rate, reverse=True):
        if len(selected) >= top_k:
            break
        if any(abs(candidate.rate - kept.rate) <= rate_tol for kept in selected):
            # Keep one representative of a near-identical screened basin.
            continue
        selected.append(candidate)
    if not selected and candidates:
        selected.append(max(candidates, key=lambda item: item.rate))
    return selected


def candidate_to_detail(
    candidate: SearchCandidate,
    m: int,
    loss_db: float,
    eta: float,
    spacing: float,
    reference_spacing: float,
    spacing_scale: float,
    outcomes: int,
    previous_rate: float,
) -> dict[str, object]:
    improvement = candidate.rate - previous_rate
    relative = 100.0 * improvement / previous_rate if previous_rate > povm.EPS else 0.0
    return {
        "M": m,
        "loss_db": fmt(loss_db),
        "eta": fmt(eta),
        "spacing_d": fmt(spacing),
        "reference_spacing_d": fmt(reference_spacing),
        "spacing_scale": fmt(spacing_scale),
        "outcomes": outcomes,
        "stage": candidate.stage,
        "source_name": candidate.source_name,
        "start_name": candidate.start_name,
        "rate": fmt(candidate.rate),
        "rate_delta_vs_previous_ykl": fmt(improvement),
        "relative_gain_vs_previous_percent": fmt(relative),
        "success_probability": fmt(candidate.success_probability),
        "useful_outcomes": candidate.useful_outcomes,
        "iterations": candidate.iterations,
        "gradient_norm": fmt(candidate.gradient_norm),
        "seconds": f"{candidate.seconds:.3f}",
    }


def summary_to_dict(summary: SearchSummary) -> dict[str, object]:
    best = summary.best
    improvement = best.rate - summary.previous_ykl_rate
    relative = (
        100.0 * improvement / summary.previous_ykl_rate
        if summary.previous_ykl_rate > povm.EPS
        else 0.0
    )
    return {
        "M": summary.m,
        "loss_db": fmt(summary.loss_db),
        "eta": fmt(summary.eta),
        "spacing_d": fmt(summary.spacing),
        "reference_spacing_d": fmt(summary.reference_spacing),
        "spacing_scale": fmt(summary.spacing_scale),
        "outcomes": summary.outcomes,
        "previous_ykl_rate": fmt(summary.previous_ykl_rate),
        "best_rate": fmt(best.rate),
        "best_minus_previous_ykl": fmt(improvement),
        "relative_gain_vs_previous_percent": fmt(relative),
        "best_source": best.source_name,
        "best_start_name": best.start_name,
        "best_polish_name": best.stage,
        "screen_starts": summary.screen_count,
        "polished_candidates": summary.polish_count,
        "screen_best_rate": fmt(summary.screen_best_rate),
        "screen_ykl_rate": fmt(summary.screen_ykl_rate),
        "polish_ykl_rate": fmt(summary.polish_ykl_rate),
        "best_minus_screen_ykl": fmt(best.rate - summary.screen_ykl_rate),
        "best_minus_polish_ykl": fmt(best.rate - summary.polish_ykl_rate),
        "near_best_screen_count": summary.near_best_screen_count,
        "near_best_polish_count": summary.near_best_polish_count,
        "near_best_abs_tol": fmt(summary.near_best_abs_tol),
        "best_gradient_norm": fmt(best.gradient_norm),
        "best_iterations": best.iterations,
        "seconds": f"{summary.seconds:.3f}",
        "best_matrix_path": str(summary.best_matrix_path),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_best_matrix(
    path: Path,
    candidate: SearchCandidate,
    m: int,
    loss_db: float,
    eta: float,
    spacing: float,
    reference_spacing: float,
    spacing_scale: float,
    outcomes: int,
    previous_rate: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        A=candidate.a,
        M=np.array(m),
        loss_db=np.array(loss_db),
        eta=np.array(eta),
        spacing_d=np.array(spacing),
        reference_spacing_d=np.array(reference_spacing),
        spacing_scale=np.array(spacing_scale),
        outcomes=np.array(outcomes),
        previous_ykl_rate=np.array(previous_rate),
        best_rate=np.array(candidate.rate),
        best_source_name=np.array(candidate.source_name),
        best_start_name=np.array(candidate.start_name),
        gradient_norm=np.array(candidate.gradient_norm),
        iterations=np.array(candidate.iterations),
    )


def run_search_one(
    m: int,
    loss_db: float,
    spacing: float,
    reference_spacing: float,
    spacing_scale: float,
    previous_rate: float,
    outcomes: int,
    random_starts: int,
    ykl_perturbations: int,
    perturb_scale: float,
    screen_iters: int,
    polish_iters: int,
    polish_top: int,
    grad_tol: float,
    initial_step: float,
    top_unique_rate_tol: float,
    near_best_abs_tol: float,
    rng: np.random.Generator,
    outdir: Path,
) -> tuple[SearchSummary, list[SearchCandidate]]:
    start_time = time.perf_counter()
    eta = 10.0 ** (-loss_db / 10.0)
    problem = povm.build_problem(loss_db, spacing, previous_rate)
    starts = generate_screen_starts(
        problem,
        outcomes,
        random_starts=random_starts,
        ykl_perturbations=ykl_perturbations,
        perturb_scale=perturb_scale,
        rng=rng,
    )

    all_candidates: list[SearchCandidate] = []
    screen_candidates: list[SearchCandidate] = []
    for name, initial in starts:
        candidate = run_local(
            initial,
            problem,
            stage="screen",
            source_name=name,
            start_name=name,
            max_iter=screen_iters,
            grad_tol=grad_tol,
            initial_step=initial_step,
        )
        screen_candidates.append(candidate)
        all_candidates.append(candidate)
        print(
            f"screen M={m} loss={loss_db:g} outcomes={outcomes} "
            f"{name}: rate={candidate.rate:.9g}, grad={candidate.gradient_norm:.3g}"
        )

    polish_sources = unique_top_candidates(
        screen_candidates, top_k=polish_top, rate_tol=top_unique_rate_tol
    )
    ykl_screen = next(item for item in screen_candidates if item.source_name == "ykl_complete")
    if all(item.source_name != "ykl_complete" for item in polish_sources):
        polish_sources.append(ykl_screen)

    polish_candidates: list[SearchCandidate] = []
    for idx, source in enumerate(polish_sources, start=1):
        candidate = run_local(
            source.a,
            problem,
            stage="polish",
            source_name=source.source_name,
            start_name=f"polish_{idx}_from_{source.source_name}",
            max_iter=polish_iters,
            grad_tol=grad_tol,
            initial_step=initial_step,
        )
        polish_candidates.append(candidate)
        all_candidates.append(candidate)
        print(
            f"polish M={m} loss={loss_db:g} outcomes={outcomes} "
            f"{candidate.start_name}: rate={candidate.rate:.9g}, "
            f"grad={candidate.gradient_norm:.3g}"
        )

    best = max(all_candidates, key=lambda item: item.rate)
    scale_tag = f"{spacing_scale:.6g}".replace(".", "p").replace("-", "m")
    best_matrix_path = (
        outdir
        / "best_matrices"
        / f"best_M{m}_loss_{loss_db:g}_scale_{scale_tag}_outcomes_{outcomes}.npz"
    )
    save_best_matrix(
        best_matrix_path,
        best,
        m=m,
        loss_db=loss_db,
        eta=eta,
        spacing=spacing,
        reference_spacing=reference_spacing,
        spacing_scale=spacing_scale,
        outcomes=outcomes,
        previous_rate=previous_rate,
    )

    screen_best_rate = max(item.rate for item in screen_candidates)
    polish_ykl = next(
        (item for item in polish_candidates if item.source_name == "ykl_complete"),
        ykl_screen,
    )
    summary = SearchSummary(
        m=m,
        loss_db=loss_db,
        eta=eta,
        spacing=spacing,
        reference_spacing=reference_spacing,
        spacing_scale=spacing_scale,
        outcomes=outcomes,
        previous_ykl_rate=previous_rate,
        best=best,
        screen_best_rate=screen_best_rate,
        screen_ykl_rate=ykl_screen.rate,
        polish_ykl_rate=polish_ykl.rate,
        screen_count=len(screen_candidates),
        polish_count=len(polish_candidates),
        near_best_screen_count=sum(
            item.rate >= best.rate - near_best_abs_tol for item in screen_candidates
        ),
        near_best_polish_count=sum(
            item.rate >= best.rate - near_best_abs_tol for item in polish_candidates
        ),
        near_best_abs_tol=near_best_abs_tol,
        seconds=time.perf_counter() - start_time,
        best_matrix_path=best_matrix_path,
    )
    return summary, all_candidates


def plot_summary(summary_df: pd.DataFrame, path: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for (m, outcomes), sub in summary_df.groupby(["M", "outcomes"]):
        sub = sub.sort_values("loss_db")
        ax.plot(
            sub["loss_db"],
            sub["best_rate"],
            marker="o",
            linewidth=2.0,
            label=f"M={int(m)}, outcomes={int(outcomes)} best",
        )
    for m, sub in summary_df.groupby("M"):
        sub = sub.sort_values("loss_db").drop_duplicates("loss_db")
        ax.plot(
            sub["loss_db"],
            sub["previous_ykl_rate"],
            linestyle="--",
            linewidth=1.8,
            label=f"M={int(m)} saved YKL",
        )
    ax.set_xlabel("Per-arm loss to Charlie (dB)")
    ax.set_ylabel("Hashing bound (bits/attempt)")
    ax.set_title("Global-search POVM best found")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", default="4")
    parser.add_argument("--losses-db", default="0.25")
    parser.add_argument("--etas", default=None)
    parser.add_argument("--outcomes", default="default")
    parser.add_argument(
        "--spacing-scales",
        default="1.0",
        help="Comma-separated multipliers applied to the saved reference spacing d.",
    )
    parser.add_argument("--random-starts", type=int, default=32)
    parser.add_argument("--ykl-perturbations", type=int, default=8)
    parser.add_argument("--perturb-scale", type=float, default=0.05)
    parser.add_argument("--screen-iters", type=int, default=80)
    parser.add_argument("--polish-iters", type=int, default=400)
    parser.add_argument("--polish-top", type=int, default=8)
    parser.add_argument("--grad-tol", type=float, default=1.0e-6)
    parser.add_argument("--initial-step", type=float, default=0.08)
    parser.add_argument("--top-unique-rate-tol", type=float, default=2.0e-4)
    parser.add_argument("--near-best-abs-tol", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path(
            "results/refined_qam_branches/"
            "qam_refined_global_rate_vs_loss_combined_dense_transitions.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/global_povm_search"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    losses = povm.parse_loss_grid(args.losses_db, args.etas)
    ms = parse_int_list(args.ms)
    rng = np.random.default_rng(args.seed)
    spacing_scales = [float(item.strip()) for item in args.spacing_scales.split(",") if item.strip()]

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    summary_path = args.outdir / "global_povm_search_summary.csv"
    detail_path = args.outdir / "global_povm_search_start_details.csv"

    for m in ms:
        set_m(m)
        previous = load_previous(args.previous, m, losses)
        outcome_counts = parse_outcomes_list(args.outcomes, m)
        for loss in losses:
            reference_spacing, previous_rate = previous[round(loss, 10)]
            for spacing_scale in spacing_scales:
                spacing = reference_spacing * spacing_scale
                for outcomes in outcome_counts:
                    summary, candidates = run_search_one(
                        m=m,
                        loss_db=loss,
                        spacing=spacing,
                        reference_spacing=reference_spacing,
                        spacing_scale=spacing_scale,
                        previous_rate=previous_rate,
                        outcomes=outcomes,
                        random_starts=args.random_starts,
                        ykl_perturbations=args.ykl_perturbations,
                        perturb_scale=args.perturb_scale,
                        screen_iters=args.screen_iters,
                        polish_iters=args.polish_iters,
                        polish_top=args.polish_top,
                        grad_tol=args.grad_tol,
                        initial_step=args.initial_step,
                        top_unique_rate_tol=args.top_unique_rate_tol,
                        near_best_abs_tol=args.near_best_abs_tol,
                        rng=rng,
                        outdir=args.outdir,
                    )
                    summary_rows.append(summary_to_dict(summary))
                    for candidate in candidates:
                        detail_rows.append(
                            candidate_to_detail(
                                candidate,
                                m=m,
                                loss_db=loss,
                                eta=summary.eta,
                                spacing=spacing,
                                reference_spacing=reference_spacing,
                                spacing_scale=spacing_scale,
                                outcomes=outcomes,
                                previous_rate=previous_rate,
                            )
                        )
                    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
                    write_csv(detail_path, DETAIL_FIELDS, detail_rows)

    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv(detail_path, DETAIL_FIELDS, detail_rows)
    summary_df = pd.read_csv(summary_path)
    plot_summary(summary_df, args.outdir / "global_povm_search_summary.png")
    print(f"Wrote {summary_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {args.outdir / 'global_povm_search_summary.png'}")


if __name__ == "__main__":
    main()
