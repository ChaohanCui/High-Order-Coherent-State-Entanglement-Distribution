#!/usr/bin/env python3
"""Numerically optimize a general C-only POVM for QAM hashing rate.

The saved QAM curves use the vacuum-omitting Bell/coset square-root/YKL
receiver.  This script asks a more direct question: if Charlie can use an
arbitrary rank-one POVM on the received optical span, how much can the same
hashing-bound objective improve?

This is a nonconvex optimization over POVMs.  The result is therefore a
numerical lower bound on the best C-only POVM, not a global-optimality proof.
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
import scipy.linalg

from mpsk_ghz_hashing import (
    EPS,
    entropy_bits,
    povm_scale,
    sparse_measurement_coefficients,
    ykl_square_root_measurement,
)
from qam_hashing import (
    coherent_pair_gram_from_amplitudes,
    local_loss_coherence_from_amplitudes,
    qam_constellation,
    standard_overlaps_after_vacuum_subtraction,
)


M = 4
DIM = M * M


FIELDS = [
    "M",
    "loss_db",
    "eta",
    "spacing_d",
    "previous_ykl_rate",
    "optimized_povm_rate",
    "rate_improvement",
    "relative_improvement_percent",
    "success_probability",
    "useful_outcomes",
    "total_outcomes",
    "best_start",
    "iterations",
    "gradient_norm",
    "seconds",
]


@dataclass(frozen=True)
class Problem:
    loss_db: float
    eta: float
    spacing: float
    previous_rate: float
    r: np.ndarray
    pair_loss: np.ndarray
    ykl_rows: np.ndarray
    ykl_scale: float


@dataclass(frozen=True)
class OptimizeResult:
    loss_db: float
    eta: float
    spacing: float
    previous_rate: float
    rate: float
    success_probability: float
    useful_outcomes: int
    total_outcomes: int
    best_start: str
    iterations: int
    gradient_norm: float
    seconds: float


@dataclass(frozen=True)
class StartResult:
    loss_db: float
    eta: float
    spacing: float
    previous_rate: float
    outcomes: int
    start_name: str
    rate: float
    improvement: float
    success_probability: float
    useful_outcomes: int
    iterations: int
    gradient_norm: float
    seconds: float


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_loss_grid(losses_db_text: str, etas_text: str | None) -> list[float]:
    if not etas_text:
        return parse_float_list(losses_db_text)
    losses = []
    for eta in parse_float_list(etas_text):
        if not (0.0 < eta <= 1.0):
            raise ValueError(f"eta={eta:g} is outside 0 < eta <= 1.")
        losses.append(-10.0 * math.log10(eta))
    return losses


def matrix_sqrt_embedding(gram: np.ndarray, tol: float = 1.0e-11) -> np.ndarray:
    evals, evecs = scipy.linalg.eigh((gram + gram.conj().T) / 2.0)
    keep = evals > tol
    if not np.any(keep):
        raise ValueError("Received-state Gram matrix has no positive support.")
    return np.diag(np.sqrt(evals[keep])) @ evecs[:, keep].conj().T


def hermitian_log2(matrix: np.ndarray, floor: float = 1.0e-13) -> np.ndarray:
    matrix = (matrix + matrix.conj().T) / 2.0
    evals, evecs = scipy.linalg.eigh(matrix)
    evals = np.maximum(np.real(evals), floor)
    return (evecs * np.log2(evals)) @ evecs.conj().T


def subnormalized_entropy_bits(matrix: np.ndarray) -> float:
    evals = scipy.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
    evals = np.real(evals)
    evals = evals[evals > 1.0e-14]
    if evals.size == 0:
        return 0.0
    return float(-np.sum(evals * np.log2(evals)))


def complete_rows_from_ykl(ykl_rows: np.ndarray, ykl_scale: float) -> np.ndarray:
    useful = math.sqrt(ykl_scale) * ykl_rows
    residual = np.eye(useful.shape[1], dtype=complex) - useful.conj().T @ useful
    residual = (residual + residual.conj().T) / 2.0
    evals, evecs = scipy.linalg.eigh(residual)
    evals = np.clip(np.real(evals), 0.0, None)
    complement = np.diag(np.sqrt(evals)) @ evecs.conj().T
    return np.vstack([useful, complement])


def split_rows_to_count(rows: np.ndarray, n_rows: int) -> np.ndarray:
    """Split existing rank-one effects until a POVM has n_rows outcomes."""

    out = [row.copy() for row in rows]
    if n_rows < len(out):
        raise ValueError(
            f"Cannot use the YKL-complete seed with only {n_rows} outcomes; "
            f"need at least {len(out)}."
        )
    while len(out) < n_rows:
        norms = np.array([np.vdot(row, row).real for row in out])
        idx = int(np.argmax(norms))
        row = out.pop(idx)
        out.append(row / math.sqrt(2.0))
        out.append(row / math.sqrt(2.0))
    return np.vstack(out)


def random_stiefel(n_rows: int, n_cols: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(n_rows, n_cols)) + 1j * rng.normal(size=(n_rows, n_cols))
    q, r = np.linalg.qr(z, mode="reduced")
    phase = np.diag(r)
    phase = np.where(np.abs(phase) > 0, phase / np.abs(phase), 1.0)
    return q * phase.conj()[None, :]


def retraction_qr(a: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(a, mode="reduced")
    phase = np.diag(r)
    phase = np.where(np.abs(phase) > 0, phase / np.abs(phase), 1.0)
    return q * phase.conj()[None, :]


def project_stiefel_tangent(a: np.ndarray, grad: np.ndarray) -> np.ndarray:
    overlap = a.conj().T @ grad
    sym = (overlap + overlap.conj().T) / 2.0
    return grad - a @ sym


def perturb_stiefel(
    a: np.ndarray, rng: np.random.Generator, scale: float
) -> np.ndarray:
    z = rng.normal(size=a.shape) + 1j * rng.normal(size=a.shape)
    direction = project_stiefel_tangent(a, z)
    norm = np.linalg.norm(direction)
    if norm <= EPS:
        return a.copy()
    return retraction_qr(a + scale * direction / norm)


def row_kernel(row: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    overlaps = row @ r
    return np.outer(overlaps, overlaps.conj()), overlaps


def objective_and_gradient(
    a: np.ndarray,
    problem: Problem,
    gradient: bool,
) -> tuple[float, float, int, np.ndarray | None]:
    total_rate = 0.0
    success_probability = 0.0
    useful_outcomes = 0
    grad = np.zeros_like(a) if gradient else None
    identity_b = np.eye(M, dtype=complex)

    for out_idx, row in enumerate(a):
        kernel, _overlaps = row_kernel(row, problem.r)
        prob = float(np.real(np.trace(kernel)) / DIM)
        if prob <= 1.0e-14:
            continue

        tau_ab = (kernel * problem.pair_loss) / DIM
        tau_ab = (tau_ab + tau_ab.conj().T) / 2.0
        rho_ab = tau_ab / prob

        tau_a = np.zeros((M, M), dtype=complex)
        for x in range(M):
            for xp in range(M):
                tau_a[x, xp] = sum(tau_ab[x * M + y, xp * M + y] for y in range(M))
        tau_a = (tau_a + tau_a.conj().T) / 2.0
        rho_a = tau_a / prob

        coherent_information = entropy_bits(rho_a) - entropy_bits(rho_ab)
        success_probability += prob
        if coherent_information <= 0.0:
            continue

        useful_outcomes += 1
        contribution = subnormalized_entropy_bits(tau_a) - subnormalized_entropy_bits(tau_ab)
        total_rate += contribution

        if gradient:
            log_tau_ab = hermitian_log2(tau_ab)
            log_tau_a = hermitian_log2(tau_a)
            grad_tau = log_tau_ab - np.kron(log_tau_a, identity_b)
            grad_kernel = grad_tau * problem.pair_loss.conj() / DIM
            effect_grad = problem.r @ grad_kernel.T @ problem.r.conj().T
            effect_grad = (effect_grad + effect_grad.conj().T) / 2.0
            grad[out_idx] = 2.0 * (row @ effect_grad)

    return float(total_rate), float(success_probability), int(useful_outcomes), grad


def stiefel_ascent(
    initial: np.ndarray,
    problem: Problem,
    max_iter: int,
    grad_tol: float,
    initial_step: float,
) -> tuple[np.ndarray, float, float, int, int, float]:
    a = initial.copy()
    rate, success, useful, grad = objective_and_gradient(a, problem, gradient=True)
    step = initial_step
    grad_norm = 0.0

    for iteration in range(1, max_iter + 1):
        assert grad is not None
        direction = project_stiefel_tangent(a, grad)
        grad_norm = float(np.linalg.norm(direction))
        if grad_norm < grad_tol:
            break

        accepted = False
        trial_step = step
        for _ in range(18):
            trial = retraction_qr(a + trial_step * direction)
            trial_rate, trial_success, trial_useful, _ = objective_and_gradient(
                trial, problem, gradient=False
            )
            if trial_rate >= rate + 1.0e-5 * trial_step * grad_norm * grad_norm:
                a = trial
                rate = trial_rate
                success = trial_success
                useful = trial_useful
                step = min(1.0, trial_step * 1.25)
                accepted = True
                break
            trial_step *= 0.5

        if not accepted:
            break
        rate, success, useful, grad = objective_and_gradient(a, problem, gradient=True)
    else:
        iteration = max_iter

    return a, float(rate), float(success), int(useful), int(iteration), grad_norm


def load_previous(path: Path, losses: list[float]) -> dict[float, tuple[float, float]]:
    df = pd.read_csv(path)
    out = {}
    for loss in losses:
        sub = df[
            (df["M"].astype(int) == M)
            & (np.abs(df["loss_db"].astype(float) - loss) < 1.0e-10)
        ]
        if sub.empty:
            raise ValueError(f"No previous {M}-QAM result for loss={loss:g} dB.")
        row = sub.iloc[0]
        out[round(loss, 10)] = (float(row["spacing_d"]), float(row["rate"]))
    return out


def build_problem(loss_db: float, spacing: float, previous_rate: float) -> Problem:
    eta = 10.0 ** (-loss_db / 10.0)
    local_amps = qam_constellation(M, spacing)
    gram, vac_overlaps = coherent_pair_gram_from_amplitudes(local_amps, eta)
    local_loss = local_loss_coherence_from_amplitudes(local_amps, eta)
    pair_loss = np.kron(local_loss, local_loss)

    coeffs, _labels, _targets = sparse_measurement_coefficients(M, "bell")
    std_overlaps, std_gram = standard_overlaps_after_vacuum_subtraction(
        coeffs, gram, vac_overlaps
    )
    ykl_overlaps, ykl_gram = ykl_square_root_measurement(std_overlaps, std_gram)
    ykl_scale = povm_scale(ykl_gram)

    r = matrix_sqrt_embedding(gram)
    ykl_rows = ykl_overlaps @ np.linalg.pinv(r)
    mismatch = float(np.linalg.norm(ykl_rows @ r - ykl_overlaps))
    if mismatch > 1.0e-7:
        raise RuntimeError(f"Could not reconstruct YKL rows; mismatch={mismatch:g}")

    return Problem(
        loss_db=loss_db,
        eta=eta,
        spacing=spacing,
        previous_rate=previous_rate,
        r=r,
        pair_loss=pair_loss,
        ykl_rows=ykl_rows,
        ykl_scale=float(ykl_scale),
    )


def optimize_problem(
    problem: Problem,
    outcomes: int,
    starts: int,
    ykl_perturbations: int,
    perturb_scale: float,
    max_iter: int,
    grad_tol: float,
    initial_step: float,
    seed: int,
) -> tuple[OptimizeResult, list[StartResult]]:
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    if outcomes < problem.r.shape[0]:
        raise ValueError(
            f"A rank-one complete POVM needs at least {problem.r.shape[0]} outcomes."
        )
    ykl_initial = split_rows_to_count(
        complete_rows_from_ykl(problem.ykl_rows, problem.ykl_scale),
        outcomes,
    )
    n_rows, n_cols = ykl_initial.shape

    starts_to_try: list[tuple[str, np.ndarray]] = [("ykl_complete", ykl_initial)]
    for idx in range(ykl_perturbations):
        starts_to_try.append(
            (
                f"ykl_perturb_{idx + 1}",
                perturb_stiefel(ykl_initial, rng, perturb_scale),
            )
        )
    for idx in range(starts):
        starts_to_try.append((f"random_{idx + 1}", random_stiefel(n_rows, n_cols, rng)))

    best: tuple[str, float, float, int, int, float] | None = None
    start_rows: list[StartResult] = []
    for name, initial in starts_to_try:
        start_seconds = time.perf_counter()
        _a, rate, success, useful, iterations, grad_norm = stiefel_ascent(
            initial,
            problem,
            max_iter=max_iter,
            grad_tol=grad_tol,
            initial_step=initial_step,
        )
        start_rows.append(
            StartResult(
                loss_db=problem.loss_db,
                eta=problem.eta,
                spacing=problem.spacing,
                previous_rate=problem.previous_rate,
                outcomes=n_rows,
                start_name=name,
                rate=rate,
                improvement=rate - problem.previous_rate,
                success_probability=success,
                useful_outcomes=useful,
                iterations=iterations,
                gradient_norm=grad_norm,
                seconds=time.perf_counter() - start_seconds,
            )
        )
        if best is None or rate > best[1]:
            best = (name, rate, success, useful, iterations, grad_norm)
        print(
            f"loss={problem.loss_db:g} dB start={name}: "
            f"rate={rate:.9g}, useful={useful}, iters={iterations}, "
            f"grad={grad_norm:.3g}"
        )

    assert best is not None
    name, rate, success, useful, iterations, grad_norm = best
    return (
        OptimizeResult(
            loss_db=problem.loss_db,
            eta=problem.eta,
            spacing=problem.spacing,
            previous_rate=problem.previous_rate,
            rate=rate,
            success_probability=success,
            useful_outcomes=useful,
            total_outcomes=n_rows,
            best_start=name,
            iterations=iterations,
            gradient_norm=grad_norm,
            seconds=time.perf_counter() - start_time,
        ),
        start_rows,
    )


def row_to_dict(row: OptimizeResult) -> dict[str, object]:
    improvement = row.rate - row.previous_rate
    rel = 100.0 * improvement / row.previous_rate if row.previous_rate > EPS else 0.0
    return {
        "M": M,
        "loss_db": fmt(row.loss_db),
        "eta": fmt(row.eta),
        "spacing_d": fmt(row.spacing),
        "previous_ykl_rate": fmt(row.previous_rate),
        "optimized_povm_rate": fmt(row.rate),
        "rate_improvement": fmt(improvement),
        "relative_improvement_percent": fmt(rel),
        "success_probability": fmt(row.success_probability),
        "useful_outcomes": row.useful_outcomes,
        "total_outcomes": row.total_outcomes,
        "best_start": row.best_start,
        "iterations": row.iterations,
        "gradient_norm": fmt(row.gradient_norm),
        "seconds": f"{row.seconds:.3f}",
    }


def write_rows(rows: list[OptimizeResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: x.loss_db):
            writer.writerow(row_to_dict(row))


def start_row_to_dict(row: StartResult) -> dict[str, object]:
    rel = (
        100.0 * row.improvement / row.previous_rate
        if row.previous_rate > EPS
        else 0.0
    )
    return {
        "M": M,
        "loss_db": fmt(row.loss_db),
        "eta": fmt(row.eta),
        "spacing_d": fmt(row.spacing),
        "previous_ykl_rate": fmt(row.previous_rate),
        "outcomes": row.outcomes,
        "start_name": row.start_name,
        "optimized_povm_rate": fmt(row.rate),
        "rate_improvement": fmt(row.improvement),
        "relative_improvement_percent": fmt(rel),
        "success_probability": fmt(row.success_probability),
        "useful_outcomes": row.useful_outcomes,
        "iterations": row.iterations,
        "gradient_norm": fmt(row.gradient_norm),
        "seconds": f"{row.seconds:.3f}",
    }


def write_start_rows(rows: list[StartResult], path: Path) -> None:
    fields = [
        "M",
        "loss_db",
        "eta",
        "spacing_d",
        "previous_ykl_rate",
        "outcomes",
        "start_name",
        "optimized_povm_rate",
        "rate_improvement",
        "relative_improvement_percent",
        "success_probability",
        "useful_outcomes",
        "iterations",
        "gradient_norm",
        "seconds",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (x.loss_db, x.outcomes, x.start_name)):
            writer.writerow(start_row_to_dict(row))


def plot_rows(rows: list[OptimizeResult], out_path: Path) -> None:
    ordered = sorted(rows, key=lambda x: x.loss_db)
    losses = [row.loss_db for row in ordered]
    previous = [row.previous_rate for row in ordered]
    optimized = [row.rate for row in ordered]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(losses, previous, marker="o", linewidth=2.2, label="vacuum-omitting YKL")
    ax.plot(losses, optimized, marker="s", linewidth=2.2, label="optimized C-only POVM")
    ax.set_xlabel("Per-arm loss to Charlie (dB)")
    ax.set_ylabel("Hashing bound (bits/attempt)")
    ax.set_title(f"{M}-QAM: YKL receiver vs optimized POVM")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    global M, DIM

    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--losses-db", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--etas", default=None)
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path(
            "results/refined_qam_branches/"
            "qam_refined_global_rate_vs_loss_combined_dense_transitions.csv"
        ),
    )
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--ykl-perturbations", type=int, default=0)
    parser.add_argument("--perturb-scale", type=float, default=0.05)
    parser.add_argument(
        "--outcomes",
        type=int,
        default=None,
        help=(
            "Number of rank-one POVM outcomes. Default is 2*M^2, i.e. the "
            "YKL outcomes plus enough complement outcomes to complete the POVM."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--grad-tol", type=float, default=1.0e-6)
    parser.add_argument("--initial-step", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    M = args.m
    DIM = M * M
    if M <= 1 or M & (M - 1):
        raise ValueError(f"M={M} is not a power of two greater than one.")
    if args.outcomes is None:
        args.outcomes = 2 * DIM
    if args.outdir is None:
        args.outdir = Path(f"results/qam{M}_general_povm")

    args.outdir.mkdir(parents=True, exist_ok=True)
    losses = parse_loss_grid(args.losses_db, args.etas)
    previous = load_previous(args.previous, losses)

    rows: list[OptimizeResult] = []
    start_rows: list[StartResult] = []
    csv_path = args.outdir / f"qam{M}_general_povm_vs_ykl.csv"
    starts_csv_path = args.outdir / f"qam{M}_general_povm_start_details.csv"
    for idx, loss in enumerate(losses):
        spacing, previous_rate = previous[round(loss, 10)]
        problem = build_problem(loss, spacing, previous_rate)
        row, starts_for_loss = optimize_problem(
            problem,
            outcomes=args.outcomes,
            starts=args.starts,
            ykl_perturbations=args.ykl_perturbations,
            perturb_scale=args.perturb_scale,
            max_iter=args.max_iter,
            grad_tol=args.grad_tol,
            initial_step=args.initial_step,
            seed=args.seed + 1000 * idx,
        )
        rows.append(row)
        start_rows.extend(starts_for_loss)
        write_rows(rows, csv_path)
        write_start_rows(start_rows, starts_csv_path)
        print(
            f"BEST loss={loss:g} dB: YKL={previous_rate:.9g}, "
            f"POVM={row.rate:.9g}, delta={row.rate - previous_rate:.9g}"
        )

    write_rows(rows, csv_path)
    write_start_rows(start_rows, starts_csv_path)
    plot_path = args.outdir / f"qam{M}_general_povm_vs_ykl.png"
    plot_rows(rows, plot_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {starts_csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
