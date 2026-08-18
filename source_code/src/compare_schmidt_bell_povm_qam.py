#!/usr/bin/env python3
"""Compare vacuum-omitting YKL with raw Schmidt-Bell POVMs for QAM."""

from __future__ import annotations

import argparse
import csv
import math
import os
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
import numpy as np
import scipy.linalg
import scipy.sparse

from mpsk_ghz_hashing import (
    EPS,
    LOSS_RANK_TOL,
    StrategyResult,
    evaluate_strategy_factorized_loss,
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


FIELDS = [
    "M",
    "constellation",
    "loss_db",
    "eta",
    "receiver",
    "best_spacing_d",
    "mean_photon_number",
    "hashing_bound_bits_per_attempt",
    "success_probability",
    "average_target_fidelity",
    "probability_weighted_fidelity",
    "min_coherent_information",
    "useful_outcomes",
    "povm_scale",
]


@dataclass(frozen=True)
class Row:
    m: int
    loss_db: float
    eta: float
    receiver: str
    spacing: float
    mean_photon_number: float
    result: StrategyResult
    scale: float


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def normalized_rows(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1)
    out = np.zeros_like(rows)
    keep = norms > EPS
    out[keep] = rows[keep] / norms[keep, None]
    return out


def raw_bell_codeword_overlaps(
    coeffs: scipy.sparse.spmatrix | np.ndarray,
    gram: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if scipy.sparse.issparse(coeffs):
        coeffs_conj_gram = coeffs.conjugate() @ gram
        raw_gram = np.asarray(coeffs_conj_gram @ coeffs.T)
    else:
        coeffs_conj_gram = np.conj(coeffs) @ gram
        raw_gram = coeffs_conj_gram @ coeffs.T

    raw_gram = (raw_gram + raw_gram.conj().T) / 2.0
    raw_norm = np.sqrt(np.maximum(np.real(np.diag(raw_gram)), EPS))
    overlaps = np.asarray(coeffs_conj_gram) / raw_norm[:, None]
    vec_gram = raw_gram / (raw_norm[:, None] * raw_norm[None, :])
    vec_gram = (vec_gram + vec_gram.conj().T) / 2.0
    return overlaps, vec_gram


def single_mode_gram_from_amplitudes(local_amps: np.ndarray, eta: float) -> np.ndarray:
    transmitted = np.sqrt(eta) * local_amps
    norms = np.abs(transmitted) ** 2
    gram = np.exp(
        -0.5 * norms[:, None]
        - 0.5 * norms[None, :]
        + np.outer(np.conj(transmitted), transmitted)
    )
    return (gram + gram.conj().T) / 2.0


def finite_qam_schmidt_bell_overlaps(
    local_amps: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Bell basis in the exact finite-QAM transmitted Schmidt-mode basis."""

    gram = single_mode_gram_from_amplitudes(local_amps, eta)
    evals, evecs = scipy.linalg.eigh(gram)
    order = np.argsort(np.real(evals))[::-1]
    evals = np.clip(np.real(evals[order]), 0.0, None)
    evecs = evecs[:, order]

    m = len(local_amps)
    omega = np.exp(2.0j * np.pi / m)
    sqrt_evals = np.sqrt(evals)
    overlaps = np.zeros((m * m, m * m), dtype=complex)
    evecs_conj = np.conj(evecs)
    row = 0
    for r in range(m):
        shifted = (np.arange(m) + r) % m
        shifted_modes = evecs_conj[:, shifted] * sqrt_evals[shifted][None, :]
        for s in range(m):
            phases = omega ** (-s * np.arange(m))
            left = evecs_conj * (sqrt_evals * phases)[None, :]
            overlaps[row] = (left @ shifted_modes.T).reshape(-1) / np.sqrt(m)
            row += 1
    return overlaps


def evaluate_schmidt_bell_qam_point(
    spacing: float,
    eta: float,
    m: int,
    rank_tol: float = LOSS_RANK_TOL,
) -> dict[str, tuple[StrategyResult, float]]:
    local_amps = qam_constellation(m, spacing)
    gram, vac_overlaps = coherent_pair_gram_from_amplitudes(local_amps, eta)
    local_loss = local_loss_coherence_from_amplitudes(local_amps, eta)
    coeffs, _labels, _bell_targets = sparse_measurement_coefficients(m, "bell")

    vac_overlaps_ykl_input, vac_gram = standard_overlaps_after_vacuum_subtraction(
        coeffs, gram, vac_overlaps
    )
    vac_ykl_overlaps, vac_ykl_gram = ykl_square_root_measurement(
        vac_overlaps_ykl_input, vac_gram
    )
    vac_targets = normalized_rows(vac_ykl_overlaps)

    raw_overlaps, raw_gram = raw_bell_codeword_overlaps(coeffs, gram)
    raw_targets = normalized_rows(raw_overlaps)
    raw_ykl_overlaps, raw_ykl_gram = ykl_square_root_measurement(raw_overlaps, raw_gram)
    raw_ykl_targets = normalized_rows(raw_ykl_overlaps)
    exact_overlaps = finite_qam_schmidt_bell_overlaps(local_amps, eta)
    exact_gram = np.eye(m * m, dtype=complex)
    exact_targets = normalized_rows(exact_overlaps)

    return {
        "vacuum_omit_ykl": (
            evaluate_strategy_factorized_loss(
                vac_ykl_overlaps,
                vac_ykl_gram,
                vac_targets,
                local_loss,
                m,
                rank_tol=rank_tol,
            ),
            povm_scale(vac_ykl_gram),
        ),
        "schmidt_bell_projection": (
            evaluate_strategy_factorized_loss(
                raw_overlaps,
                raw_gram,
                raw_targets,
                local_loss,
                m,
                rank_tol=rank_tol,
            ),
            povm_scale(raw_gram),
        ),
        "schmidt_bell_srm": (
            evaluate_strategy_factorized_loss(
                raw_ykl_overlaps,
                raw_ykl_gram,
                raw_ykl_targets,
                local_loss,
                m,
                rank_tol=rank_tol,
            ),
            povm_scale(raw_ykl_gram),
        ),
        "finite_qam_schmidt_bell": (
            evaluate_strategy_factorized_loss(
                exact_overlaps,
                exact_gram,
                exact_targets,
                local_loss,
                m,
                rank_tol=rank_tol,
            ),
            1.0,
        ),
    }


def optimize_receivers_for_loss(
    m: int,
    loss_db: float,
    spacing_min: float,
    spacing_max: float,
    coarse_points: int,
    refine_points: int,
    rank_tol: float,
) -> list[Row]:
    eta = 10.0 ** (-loss_db / 10.0)
    cache: dict[float, dict[str, tuple[StrategyResult, float]]] = {}

    def evaluate(spacing: float) -> dict[str, tuple[StrategyResult, float]]:
        key = round(float(spacing), 12)
        if key not in cache:
            cache[key] = evaluate_schmidt_bell_qam_point(float(spacing), eta, m, rank_tol)
        return cache[key]

    coarse = np.linspace(spacing_min, spacing_max, coarse_points)
    for spacing in coarse:
        evaluate(float(spacing))

    receivers = [
        "vacuum_omit_ykl",
        "schmidt_bell_projection",
        "schmidt_bell_srm",
        "finite_qam_schmidt_bell",
    ]
    best: dict[str, tuple[float, StrategyResult, float]] = {}
    for receiver in receivers:
        candidates = [
            (float(spacing), *evaluate(float(spacing))[receiver]) for spacing in coarse
        ]
        best[receiver] = max(candidates, key=lambda item: item[1].rate)

    step = float(coarse[1] - coarse[0]) if coarse_points > 1 else 0.1
    for receiver in receivers:
        spacing0 = best[receiver][0]
        lo = max(spacing_min, spacing0 - 2.0 * step)
        hi = min(spacing_max, spacing0 + 2.0 * step)
        if math.isclose(spacing0, spacing_min):
            hi = min(spacing_max, spacing0 + 4.0 * step)
        if math.isclose(spacing0, spacing_max):
            lo = max(spacing_min, spacing0 - 4.0 * step)
        refine = np.linspace(lo, hi, refine_points)
        for spacing in refine:
            evaluate(float(spacing))
        candidates = [
            best[receiver],
            *[(float(spacing), *evaluate(float(spacing))[receiver]) for spacing in refine],
        ]
        best[receiver] = max(candidates, key=lambda item: item[1].rate)

    rows = []
    for receiver, (spacing, result, scale) in best.items():
        local_amps = qam_constellation(m, spacing)
        rows.append(
            Row(
                m=m,
                loss_db=loss_db,
                eta=eta,
                receiver=receiver,
                spacing=spacing,
                mean_photon_number=float(np.mean(np.abs(local_amps) ** 2)),
                result=result,
                scale=scale,
            )
        )
    return rows


def row_to_dict(row: Row) -> dict[str, object]:
    result = row.result
    return {
        "M": row.m,
        "constellation": f"{row.m}-QAM",
        "loss_db": fmt(row.loss_db),
        "eta": fmt(row.eta),
        "receiver": row.receiver,
        "best_spacing_d": fmt(row.spacing),
        "mean_photon_number": fmt(row.mean_photon_number),
        "hashing_bound_bits_per_attempt": fmt(result.rate),
        "success_probability": fmt(result.success_probability),
        "average_target_fidelity": fmt(result.average_fidelity),
        "probability_weighted_fidelity": fmt(
            result.success_probability * result.average_fidelity
        ),
        "min_coherent_information": fmt(result.min_coherent_information),
        "useful_outcomes": result.useful_outcomes,
        "povm_scale": fmt(row.scale),
    }


def write_rows(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.m, item.loss_db, item.receiver)):
            writer.writerow(row_to_dict(row))


def plot_rows(rows: list[Row], path: Path) -> None:
    data = [row_to_dict(row) for row in rows]
    import pandas as pd

    df = pd.DataFrame(data)
    df["M"] = df["M"].astype(int)
    df["loss_db"] = df["loss_db"].astype(float)
    df["hashing_bound_bits_per_attempt"] = df[
        "hashing_bound_bits_per_attempt"
    ].astype(float)
    colors = {
        "vacuum_omit_ykl": "#333333",
        "schmidt_bell_projection": "#C44E52",
        "schmidt_bell_srm": "#4C78A8",
        "finite_qam_schmidt_bell": "#2F9E44",
    }
    linestyles = {
        "vacuum_omit_ykl": "-",
        "schmidt_bell_projection": ":",
        "schmidt_bell_srm": "--",
        "finite_qam_schmidt_bell": "-.",
    }

    fig, ax = plt.subplots(figsize=(5.6, 3.5), constrained_layout=True)
    for receiver in colors:
        sub = df[df["receiver"] == receiver].sort_values("loss_db")
        ax.plot(
            sub["loss_db"],
            sub["hashing_bound_bits_per_attempt"],
            color=colors[receiver],
            linestyle=linestyles[receiver],
            marker="o",
            markersize=3.0,
            linewidth=1.8,
            label=receiver.replace("_", " "),
        )
    ax.set_title(f"{int(df['M'].iloc[0])}-QAM Schmidt-Bell receiver check")
    ax.set_xlabel("loss to Charlie (dB)")
    ax.set_ylabel("optimized hashing bound")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results/schmidt_bell_povm_qam"))
    parser.add_argument("--ms", default="8")
    parser.add_argument("--losses-db", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--spacing-min", type=float, default=0.05)
    parser.add_argument("--spacing-max", type=float, default=5.0)
    parser.add_argument("--coarse-points", type=int, default=41)
    parser.add_argument("--refine-points", type=int, default=41)
    parser.add_argument("--rank-tol", type=float, default=LOSS_RANK_TOL)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    ms = parse_int_list(args.ms)
    losses = parse_float_list(args.losses_db)
    jobs = [(m, loss) for m in ms for loss in losses]

    rows: list[Row] = []
    summary = args.outdir / "schmidt_bell_povm_qam_comparison.csv"

    if args.workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    optimize_receivers_for_loss,
                    m,
                    loss,
                    args.spacing_min,
                    args.spacing_max,
                    args.coarse_points,
                    args.refine_points,
                    args.rank_tol,
                ): (m, loss)
                for m, loss in jobs
            }
            for future in as_completed(futures):
                batch = future.result()
                rows.extend(batch)
                write_rows(rows, summary)
                m, loss = futures[future]
                base = next(row for row in batch if row.receiver == "vacuum_omit_ykl")
                best_schmidt = max(
                    (
                        row
                        for row in batch
                        if row.receiver.startswith("schmidt")
                        or row.receiver == "finite_qam_schmidt_bell"
                    ),
                    key=lambda row: row.result.rate,
                )
                print(
                    f"M={m} loss={loss:g} dB: vacuumYKL={base.result.rate:.6g}, "
                    f"bestSchmidt={best_schmidt.result.rate:.6g}, "
                    f"d={best_schmidt.spacing:.6g}"
                )
    else:
        for m, loss in jobs:
            batch = optimize_receivers_for_loss(
                m,
                loss,
                args.spacing_min,
                args.spacing_max,
                args.coarse_points,
                args.refine_points,
                args.rank_tol,
            )
            rows.extend(batch)
            write_rows(rows, summary)
            base = next(row for row in batch if row.receiver == "vacuum_omit_ykl")
            best_schmidt = max(
                (
                    row
                    for row in batch
                    if row.receiver.startswith("schmidt")
                    or row.receiver == "finite_qam_schmidt_bell"
                ),
                key=lambda row: row.result.rate,
            )
            print(
                f"M={m} loss={loss:g} dB: vacuumYKL={base.result.rate:.6g}, "
                f"bestSchmidt={best_schmidt.result.rate:.6g}, "
                f"d={best_schmidt.spacing:.6g}"
            )

    write_rows(rows, summary)
    plot_rows(rows, args.outdir / "schmidt_bell_povm_qam_comparison.png")
    print(summary)
    print(args.outdir / "schmidt_bell_povm_qam_comparison.png")


if __name__ == "__main__":
    main()
