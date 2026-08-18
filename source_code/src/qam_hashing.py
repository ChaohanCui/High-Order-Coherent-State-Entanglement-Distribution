#!/usr/bin/env python3
"""Coherent-state assisted rectangular QAM entanglement distribution sweeps.

This script is intentionally separate from mpsk_ghz_hashing.py.  It keeps the
same memory-target and POVM machinery, but replaces the phase-only PSK alphabet
with a rectangular QAM alphabet whose nearest-neighbor coherent-amplitude
spacing is optimized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

from mpsk_ghz_hashing import (
    EPS,
    LOSS_RANK_TOL,
    StrategyResult,
    evaluate_strategy_factorized_loss,
    local_loss_coherence_from_amplitudes,
    sparse_measurement_coefficients,
    ykl_square_root_measurement,
)


@dataclass(frozen=True)
class QamSweepBest:
    family: str
    m: int
    rows: int
    cols: int
    loss_db: float
    eta: float
    strategy: str
    best_spacing: float
    mean_photon_number: float
    result: StrategyResult


def qam_shape(m: int) -> tuple[int, int]:
    if m <= 1 or m & (m - 1):
        raise ValueError(f"M={m} is not a power of two greater than one.")
    nbits = int(round(math.log2(m)))
    row_bits = nbits // 2
    col_bits = nbits - row_bits
    return 2**row_bits, 2**col_bits


def sign_from_bit(bit: int) -> float:
    return 1.0 if bit == 0 else -1.0


def qam_constellation(m: int, spacing: float) -> np.ndarray:
    """Return DK-natural rectangular QAM amplitudes with neighbor spacing d.

    Binary labels are split into real-axis bits followed by imaginary-axis
    bits.  With s_k = +1 for bit 0 and -1 for bit 1, the coordinate along each
    axis is the signed binary-weighted sum, multiplied by d/2.  This gives
    8-QAM as a 2 by 4 grid, 16-QAM as 4 by 4, and 32-QAM as 4 by 8.
    """

    nbits = int(round(math.log2(m)))
    rows, cols = qam_shape(m)
    col_bits = int(round(math.log2(cols)))
    row_bits = int(round(math.log2(rows)))
    amps = np.empty(m, dtype=complex)
    half = spacing / 2.0
    real_weights = [2.0 ** (col_bits - 1 - k) for k in range(col_bits)]
    imag_weights = [2.0 ** (row_bits - 1 - k) for k in range(row_bits)]
    for label in range(m):
        bits = [(label >> (nbits - 1 - k)) & 1 for k in range(nbits)]
        signs = [sign_from_bit(bit) for bit in bits]
        real = sum(weight * signs[k] for k, weight in enumerate(real_weights))
        imag = sum(
            weight * signs[col_bits + k] for k, weight in enumerate(imag_weights)
        )
        amps[label] = half * (real + 1j * imag)
    return amps


def pair_amplitude_arrays(local_amps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = len(local_amps)
    amp_a = np.empty(m * m, dtype=complex)
    amp_b = np.empty(m * m, dtype=complex)
    for a in range(m):
        for b in range(m):
            idx = a * m + b
            amp_a[idx] = local_amps[a]
            amp_b[idx] = local_amps[b]
    return amp_a, amp_b


def coherent_pair_gram_from_amplitudes(
    local_amps: np.ndarray, eta: float
) -> tuple[np.ndarray, np.ndarray]:
    amp_a, amp_b = pair_amplitude_arrays(np.sqrt(eta) * local_amps)
    norms = np.abs(amp_a) ** 2 + np.abs(amp_b) ** 2
    gram = np.exp(
        -0.5 * norms[:, None]
        - 0.5 * norms[None, :]
        + np.outer(np.conj(amp_a), amp_a)
        + np.outer(np.conj(amp_b), amp_b)
    )
    gram = (gram + gram.conj().T) / 2.0
    vac_overlaps = np.exp(-0.5 * norms)
    return gram, vac_overlaps


def loss_coherence_matrix_from_amplitudes(local_amps: np.ndarray, eta: float) -> np.ndarray:
    amp_a, amp_b = pair_amplitude_arrays(np.sqrt(1.0 - eta) * local_amps)
    norms = np.abs(amp_a) ** 2 + np.abs(amp_b) ** 2
    loss = np.exp(
        -0.5 * norms[:, None]
        - 0.5 * norms[None, :]
        + np.outer(amp_a, np.conj(amp_a))
        + np.outer(amp_b, np.conj(amp_b))
    )
    return loss


def standard_overlaps_after_vacuum_subtraction(
    coeffs: np.ndarray, gram: np.ndarray, vac_overlaps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute <phi_k|optical_j> and <phi_k|phi_l> for QAM standard vectors."""

    if scipy.sparse.issparse(coeffs):
        coeffs_conj_gram = coeffs.conjugate() @ gram
        raw_gram = np.asarray(coeffs_conj_gram @ coeffs.T)
        raw_norm2 = np.real(np.diag(raw_gram))
    else:
        coeffs_conj_gram = np.conj(coeffs) @ gram
        raw_gram = coeffs_conj_gram @ coeffs.T
        raw_norm2 = np.real(np.diag(raw_gram))
    raw_norm = np.sqrt(np.maximum(np.real(raw_norm2), EPS))

    psi_overlaps = coeffs_conj_gram / raw_norm[:, None]
    vac_overlap = np.asarray(coeffs @ vac_overlaps).ravel() / raw_norm
    residual_norm2 = np.maximum(1.0 - np.abs(vac_overlap) ** 2, EPS)
    residual_norm = np.sqrt(residual_norm2)

    overlaps = (
        psi_overlaps - np.conj(vac_overlap)[:, None] * vac_overlaps[None, :]
    ) / residual_norm[:, None]

    psi_gram = raw_gram / (raw_norm[:, None] * raw_norm[None, :])
    vec_gram = (
        psi_gram - np.conj(vac_overlap)[:, None] * vac_overlap[None, :]
    ) / (residual_norm[:, None] * residual_norm[None, :])
    vec_gram = (vec_gram + vec_gram.conj().T) / 2.0
    return overlaps, vec_gram


def evaluate_point(
    spacing: float,
    eta: float,
    m: int,
    family: str,
    rank_tol: float = LOSS_RANK_TOL,
) -> dict[str, StrategyResult]:
    local_amps = qam_constellation(m, spacing)
    coeffs, _labels, targets = sparse_measurement_coefficients(m, family)
    gram, vac_overlaps = coherent_pair_gram_from_amplitudes(local_amps, eta)
    local_loss = local_loss_coherence_from_amplitudes(local_amps, eta)
    std_overlaps, std_gram = standard_overlaps_after_vacuum_subtraction(
        coeffs, gram, vac_overlaps
    )
    ykl_overlaps, ykl_gram = ykl_square_root_measurement(std_overlaps, std_gram)
    return {
        "Standard": evaluate_strategy_factorized_loss(
            std_overlaps, std_gram, targets, local_loss, m, rank_tol=rank_tol
        ),
        "YKL": evaluate_strategy_factorized_loss(
            ykl_overlaps, ykl_gram, targets, local_loss, m, rank_tol=rank_tol
        ),
    }


def _evaluate_spacing_task(
    args: tuple[float, str, int, float, float]
) -> tuple[float, dict[str, StrategyResult]]:
    spacing, family, m, eta, rank_tol = args
    key = round(float(spacing), 12)
    return key, evaluate_point(float(spacing), eta, m, family, rank_tol=rank_tol)


def optimize_spacing(
    family: str,
    m: int,
    loss_db: float,
    spacing_min: float,
    spacing_max: float,
    coarse_points: int,
    refine_points: int,
    executor: Executor | None = None,
    rank_tol: float = LOSS_RANK_TOL,
) -> dict[str, QamSweepBest]:
    eta = 10.0 ** (-loss_db / 10.0)
    coarse_spacings = np.linspace(spacing_min, spacing_max, coarse_points)
    cache: dict[float, dict[str, StrategyResult]] = {}

    def evaluate_spacings(spacings: Iterable[float]) -> None:
        missing = []
        for spacing in spacings:
            key = round(float(spacing), 12)
            if key not in cache:
                missing.append(float(spacing))
        if not missing:
            return
        if executor is None or len(missing) == 1:
            for spacing in missing:
                key, result = _evaluate_spacing_task(
                    (spacing, family, m, eta, rank_tol)
                )
                cache[key] = result
        else:
            tasks = [(spacing, family, m, eta, rank_tol) for spacing in missing]
            for key, result in executor.map(_evaluate_spacing_task, tasks):
                cache[key] = result

    def eval_spacing(spacing: float) -> dict[str, StrategyResult]:
        key = round(float(spacing), 12)
        if key not in cache:
            evaluate_spacings([spacing])
        return cache[key]

    evaluate_spacings(coarse_spacings)

    best: dict[str, tuple[float, StrategyResult]] = {}
    for strategy in ["Standard", "YKL"]:
        pairs = [
            (spacing, eval_spacing(float(spacing))[strategy])
            for spacing in coarse_spacings
        ]
        best[strategy] = max(pairs, key=lambda item: item[1].rate)

    step = float(coarse_spacings[1] - coarse_spacings[0]) if coarse_points > 1 else 0.1
    for strategy, (spacing0, _result0) in list(best.items()):
        lo = max(spacing_min, spacing0 - 2.0 * step)
        hi = min(spacing_max, spacing0 + 2.0 * step)
        if math.isclose(spacing0, spacing_min):
            hi = min(spacing_max, spacing0 + 4.0 * step)
        if math.isclose(spacing0, spacing_max):
            lo = max(spacing_min, spacing0 - 4.0 * step)
        refine_spacings = np.linspace(lo, hi, refine_points)
        evaluate_spacings(refine_spacings)
        pairs = [
            (spacing, eval_spacing(float(spacing))[strategy])
            for spacing in refine_spacings
        ]
        best[strategy] = max([best[strategy], *pairs], key=lambda item: item[1].rate)

    rows, cols = qam_shape(m)
    return {
        strategy: QamSweepBest(
            family=family,
            m=m,
            rows=rows,
            cols=cols,
            loss_db=loss_db,
            eta=eta,
            strategy=strategy,
            best_spacing=float(spacing),
            mean_photon_number=float(np.mean(np.abs(qam_constellation(m, spacing)) ** 2)),
            result=result,
        )
        for strategy, (spacing, result) in best.items()
    }


def run_sweep(
    family: str,
    m_values: Iterable[int],
    losses_db: Iterable[float],
    spacing_min: float,
    spacing_max: float,
    coarse_points: int,
    refine_points: int,
    workers: int = 1,
    rank_tol: float = LOSS_RANK_TOL,
) -> list[QamSweepBest]:
    rows: list[QamSweepBest] = []
    executor_cm = ThreadPoolExecutor(max_workers=workers) if workers and workers > 1 else None
    try:
        if executor_cm is not None:
            executor_cm.__enter__()
        for m in m_values:
            for loss_db in losses_db:
                print(f"Running family={family}, M={m}-QAM, loss={loss_db:.2f} dB")
                best = optimize_spacing(
                    family=family,
                    m=m,
                    loss_db=loss_db,
                    spacing_min=spacing_min,
                    spacing_max=spacing_max,
                    coarse_points=coarse_points,
                    refine_points=refine_points,
                    executor=executor_cm,
                    rank_tol=rank_tol,
                )
                rows.extend(best[strategy] for strategy in ["Standard", "YKL"])
                for strategy in ["Standard", "YKL"]:
                    item = best[strategy]
                    print(
                        f"  {strategy:8s}: R={item.result.rate:.6f} bits, "
                        f"d={item.best_spacing:.3f}, "
                        f"nbar={item.mean_photon_number:.3f}, "
                        f"Psucc={item.result.success_probability:.4f}, "
                        f"Favg={item.result.average_fidelity:.4f}"
                    )
    finally:
        if executor_cm is not None:
            executor_cm.__exit__(None, None, None)
    return rows


def write_csv(rows: list[QamSweepBest], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "M",
                "constellation",
                "family",
                "qubits_per_side",
                "rows",
                "cols",
                "loss_db",
                "eta",
                "strategy",
                "best_spacing",
                "mean_photon_number",
                "hashing_bound_bits_per_attempt",
                "success_probability",
                "average_target_fidelity",
                "min_coherent_information",
                "useful_outcomes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.m,
                    f"{row.m}-QAM",
                    row.family,
                    int(round(math.log2(row.m))),
                    row.rows,
                    row.cols,
                    f"{row.loss_db:.6g}",
                    f"{row.eta:.12g}",
                    row.strategy,
                    f"{row.best_spacing:.12g}",
                    f"{row.mean_photon_number:.12g}",
                    f"{row.result.rate:.12g}",
                    f"{row.result.success_probability:.12g}",
                    f"{row.result.average_fidelity:.12g}",
                    f"{row.result.min_coherent_information:.12g}",
                    row.result.useful_outcomes,
                ]
            )


def write_json(rows: list[QamSweepBest], path: Path) -> None:
    data = []
    for row in rows:
        data.append(
            {
                "M": row.m,
                "constellation": f"{row.m}-QAM",
                "family": row.family,
                "qubits_per_side": int(round(math.log2(row.m))),
                "rows": row.rows,
                "cols": row.cols,
                "loss_db": row.loss_db,
                "eta": row.eta,
                "strategy": row.strategy,
                "best_spacing": row.best_spacing,
                "mean_photon_number": row.mean_photon_number,
                "hashing_bound_bits_per_attempt": row.result.rate,
                "success_probability": row.result.success_probability,
                "average_target_fidelity": row.result.average_fidelity,
                "min_coherent_information": row.result.min_coherent_information,
                "useful_outcomes": row.result.useful_outcomes,
            }
        )
    path.write_text(json.dumps(data, indent=2))


def plot_qam_rates(rows: list[QamSweepBest], strategy: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colors = {8: "#D65A31", 16: "#7B2CBF", 32: "#0B7285"}
    markers = {8: "^", 16: "D", 32: "P"}
    subset_all = [row for row in rows if row.strategy == strategy]
    for m in sorted({row.m for row in subset_all}):
        subset = sorted([row for row in subset_all if row.m == m], key=lambda r: r.loss_db)
        ax.plot(
            [row.loss_db for row in subset],
            [row.result.rate for row in subset],
            marker=markers.get(m, "o"),
            linestyle="-",
            color=colors.get(m, "#333333"),
            label=f"{m}-QAM",
            linewidth=1.9,
            markersize=4.5,
        )
    ax.set_xlabel("Per-arm channel loss to Charlie (dB)")
    ax.set_ylabel("Optimized weighted hashing bound (bits/attempt)")
    positive_rates = [row.result.rate for row in subset_all if row.result.rate > 0.0]
    if positive_rates:
        ax.set_yscale("log")
        ax.set_ylim(min(positive_rates) * 0.75, max(positive_rates) * 1.25)
    ax.set_xlim(-0.05, 3.05)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    ax.set_title(f"Rectangular QAM, {strategy} POVM")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_best_spacing(rows: list[QamSweepBest], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    colors = {8: "#D65A31", 16: "#7B2CBF", 32: "#0B7285"}
    markers = {"Standard": "o", "YKL": "s"}
    linestyles = {"Standard": "--", "YKL": "-"}
    for m in sorted({row.m for row in rows}):
        for strategy in ["Standard", "YKL"]:
            subset = sorted(
                [row for row in rows if row.m == m and row.strategy == strategy],
                key=lambda r: r.loss_db,
            )
            ax.plot(
                [row.loss_db for row in subset],
                [row.best_spacing for row in subset],
                marker=markers[strategy],
                linestyle=linestyles[strategy],
                color=colors.get(m, "#333333"),
                label=f"{m}-QAM {strategy}",
                linewidth=1.8,
                markersize=4.5,
            )
    ax.set_xlabel("Per-arm channel loss to Charlie (dB)")
    ax.set_ylabel("Optimized QAM nearest-neighbor spacing d")
    ax.set_xlim(-0.05, 3.05)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def read_psk_csv(path: Path, allowed_m: set[int]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            m = int(row["M"])
            if m not in allowed_m:
                continue
            out.append(
                {
                    "M": m,
                    "modulation": "PSK",
                    "strategy": row["strategy"],
                    "loss_db": float(row["loss_db"]),
                    "rate": float(row["hashing_bound_bits_per_attempt"]),
                }
            )
    return out


def qam_rows_for_comparison(
    rows: list[QamSweepBest], allowed_m: set[int]
) -> list[dict[str, object]]:
    return [
        {
            "M": row.m,
            "modulation": "QAM",
            "strategy": row.strategy,
            "loss_db": row.loss_db,
            "rate": row.result.rate,
        }
        for row in rows
        if row.m in allowed_m
    ]


def plot_psk_qam_comparison(
    rows: list[dict[str, object]], strategy: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    colors = {
        (8, "PSK"): "#D65A31",
        (8, "QAM"): "#B43E8F",
        (16, "PSK"): "#7B2CBF",
        (16, "QAM"): "#0B7285",
        (32, "QAM"): "#495057",
    }
    markers = {"PSK": "o", "QAM": "s"}
    linestyles = {"PSK": "-", "QAM": "--"}
    subset_all = [row for row in rows if row["strategy"] == strategy]
    for m in sorted({int(row["M"]) for row in subset_all}):
        for modulation in ["PSK", "QAM"]:
            subset = sorted(
                [
                    row
                    for row in subset_all
                    if row["M"] == m and row["modulation"] == modulation
                ],
                key=lambda r: float(r["loss_db"]),
            )
            if not subset:
                continue
            ax.plot(
                [float(row["loss_db"]) for row in subset],
                [float(row["rate"]) for row in subset],
                marker=markers[modulation],
                linestyle=linestyles[modulation],
                color=colors.get((m, modulation), "#333333"),
                label=f"{m}-{modulation}",
                linewidth=1.9,
                markersize=4.2,
            )
    ax.set_xlabel("Per-arm channel loss to Charlie (dB)")
    ax.set_ylabel("Optimized weighted hashing bound (bits/attempt)")
    positive_rates = [float(row["rate"]) for row in subset_all if float(row["rate"]) > 0.0]
    if positive_rates:
        ax.set_yscale("log")
        ax.set_ylim(min(positive_rates) * 0.75, max(positive_rates) * 1.25)
    ax.set_xlim(-0.05, 3.05)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    ax.set_title(f"Bell/coset targets, {strategy} POVM")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def latex_float(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def write_report_tables(rows: list[QamSweepBest], path: Path) -> None:
    compact_losses = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    by_key = {(row.m, round(row.loss_db, 10), row.strategy): row for row in rows}
    m_values = sorted({row.m for row in rows})

    lines: list[str] = []
    lines.append("% Auto-generated by qam_hashing.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Rectangular QAM optimized weighted hashing bound, in bits per attempt.}")
    lines.append("\\label{tab:qam-hashing-results}")
    lines.append("\\scriptsize")
    colspec = "cc" + "rr" * len(m_values)
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\hline")
    header = ["Loss (dB)", "$\\eta$"]
    for m in m_values:
        header.extend([f"{m}-QAM Std.", f"{m}-QAM YKL"])
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")
    for loss in compact_losses:
        if (m_values[0], loss, "Standard") not in by_key:
            continue
        eta = by_key[(m_values[0], loss, "Standard")].eta
        row = [latex_float(loss, 1), latex_float(eta, 4)]
        for m in m_values:
            row.append(latex_float(by_key[(m, loss, "Standard")].result.rate, 4))
            row.append(latex_float(by_key[(m, loss, "YKL")].result.rate, 4))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Optimized QAM nearest-neighbor spacing $d$ for the same sweep.}")
    lines.append("\\label{tab:qam-spacing-results}")
    lines.append("\\scriptsize")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")
    for loss in compact_losses:
        if (m_values[0], loss, "Standard") not in by_key:
            continue
        eta = by_key[(m_values[0], loss, "Standard")].eta
        row = [latex_float(loss, 1), latex_float(eta, 4)]
        for m in m_values:
            row.append(latex_float(by_key[(m, loss, "Standard")].best_spacing, 3))
            row.append(latex_float(by_key[(m, loss, "YKL")].best_spacing, 3))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="qam_hashing_results_0p1db")
    parser.add_argument("--spacing-min", type=float, default=0.1)
    parser.add_argument("--spacing-max", type=float, default=10.0)
    parser.add_argument("--coarse-points", type=int, default=45)
    parser.add_argument("--refine-points", type=int, default=31)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--loss-rank-tol",
        type=float,
        default=LOSS_RANK_TOL,
        help=(
            "Relative eigenvalue cutoff for the factorized environment-loss "
            "Gram matrix. Larger values are faster but approximate."
        ),
    )
    parser.add_argument(
        "--family",
        choices=["ghz", "bell"],
        default="bell",
        help="Measurement target family: rank-2 GHZ pairs or full M-branch Bell states.",
    )
    parser.add_argument("--losses-db", default="0,0.5,1,1.5,2,2.5,3")
    parser.add_argument("--loss-min", type=float, default=None)
    parser.add_argument("--loss-max", type=float, default=None)
    parser.add_argument("--loss-step", type=float, default=None)
    parser.add_argument("--m-values", default="8,16")
    parser.add_argument(
        "--psk-csv",
        default="mpsk_bell_hashing_results_0p1db_with16/mpsk_bell_hashing_summary.csv",
        help="Existing Bell/coset PSK CSV to use for PSK-vs-QAM comparison plots.",
    )
    args = parser.parse_args()

    thread_limiter = None
    if args.workers and args.workers > 1:
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(var, "1")
        try:
            from threadpoolctl import threadpool_limits

            thread_limiter = threadpool_limits(limits=1)
            thread_limiter.__enter__()
        except Exception:
            thread_limiter = None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.loss_min is not None or args.loss_max is not None or args.loss_step is not None:
        if args.loss_min is None or args.loss_max is None or args.loss_step is None:
            raise ValueError("--loss-min, --loss-max, and --loss-step must be set together.")
        if args.loss_step <= 0:
            raise ValueError("--loss-step must be positive.")
        count = int(round((args.loss_max - args.loss_min) / args.loss_step))
        losses_db = [
            round(args.loss_min + i * args.loss_step, 10)
            for i in range(count + 1)
            if args.loss_min + i * args.loss_step <= args.loss_max + 1.0e-9
        ]
    else:
        losses_db = [float(x) for x in args.losses_db.split(",") if x.strip()]

    m_values = [int(x) for x in args.m_values.split(",") if x.strip()]
    for m in m_values:
        qam_shape(m)

    try:
        rows = run_sweep(
            family=args.family,
            m_values=m_values,
            losses_db=losses_db,
            spacing_min=args.spacing_min,
            spacing_max=args.spacing_max,
            coarse_points=args.coarse_points,
            refine_points=args.refine_points,
            workers=args.workers,
            rank_tol=args.loss_rank_tol,
        )

        stem = f"qam_{args.family}_hashing_summary"
        write_csv(rows, outdir / f"{stem}.csv")
        write_json(rows, outdir / f"{stem}.json")
        write_report_tables(rows, outdir / "report_tables.tex")
        plot_qam_rates(rows, "Standard", outdir / "standard_hash_bound_vs_loss.png")
        plot_qam_rates(rows, "YKL", outdir / "ykl_hash_bound_vs_loss.png")
        plot_best_spacing(rows, outdir / "best_spacing_vs_loss.png")

        comparison_m_values = set(m_values)
        comparison_rows = read_psk_csv(
            Path(args.psk_csv), comparison_m_values
        ) + qam_rows_for_comparison(rows, comparison_m_values)
        if comparison_rows:
            plot_psk_qam_comparison(
                comparison_rows,
                "Standard",
                outdir / "bell_standard_psk_qam_comparison.png",
            )
            plot_psk_qam_comparison(
                comparison_rows,
                "YKL",
                outdir / "bell_ykl_psk_qam_comparison.png",
            )

        print(f"Wrote results to {outdir}")
    finally:
        if thread_limiter is not None:
            thread_limiter.__exit__(None, None, None)


if __name__ == "__main__":
    main()
