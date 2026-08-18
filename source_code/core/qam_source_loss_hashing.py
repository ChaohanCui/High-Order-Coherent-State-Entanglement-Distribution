#!/usr/bin/env python3
"""QAM hashing-bound scan with explicit sequential source-generation loss.

The older QAM simulations assume that Alice and Bob already possess an ideal
rectangular coherent-state alphabet.  This script instead generates that
alphabet by a sequential Duan-Kimble-like source model:

    displacement/phase choice -> qubit-controlled pi reflection -> source loss

for each memory qubit.  The drive amplitudes are calibrated so the outgoing
signal mode, after all source-generation loss, still lands exactly on the
requested rectangular QAM grid with nearest-neighbor spacing d.  The price is
that a branch-dependent environment mode is emitted after every controlled
reflection, and those source leakage modes are included in the conditional
memory-state entropy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

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

from mpsk_ghz_hashing import (
    EPS,
    LOSS_RANK_TOL,
    StrategyResult,
    evaluate_strategy_factorized_loss,
    sparse_measurement_coefficients,
    ykl_square_root_measurement,
)
from qam_hashing import (
    coherent_pair_gram_from_amplitudes,
    qam_constellation,
    qam_shape,
    standard_overlaps_after_vacuum_subtraction,
)


DISPLAY_NAMES = {2: "2-QAM", 4: "4-QAM", 8: "8-QAM", 16: "16-QAM"}


@dataclass(frozen=True)
class SourceLossPoint:
    m: int
    source_loss_db: float
    eta_source: float
    channel_loss_db: float
    eta_channel: float
    spacing: float
    mean_signal_photon_number: float
    mean_source_leakage_photons: float
    max_programmed_drive_photons: float
    result: StrategyResult


@dataclass(frozen=True)
class OptimizedSourceLossRow:
    m: int
    source_loss_db: float
    eta_source: float
    channel_loss_db: float
    eta_channel: float
    best_spacing: float
    mean_signal_photon_number: float
    mean_source_leakage_photons: float
    max_programmed_drive_photons: float
    result: StrategyResult


def qam_axis_coefficients(m: int, spacing: float) -> np.ndarray:
    """Complex signed-sum coefficients for the rectangular QAM grid.

    The coefficients match qam_hashing.qam_constellation: real-axis bits first,
    then imaginary-axis bits, with binary weights.  For example, 16-QAM uses
    [d, d/2, i d, i d/2].
    """

    rows, cols = qam_shape(m)
    col_bits = int(round(math.log2(cols)))
    row_bits = int(round(math.log2(rows)))
    half = spacing / 2.0
    coeffs: list[complex] = []
    coeffs.extend(
        half * 2.0 ** (col_bits - 1 - k) for k in range(col_bits)
    )
    coeffs.extend(
        1j * half * 2.0 ** (row_bits - 1 - k) for k in range(row_bits)
    )
    return np.asarray(coeffs, dtype=complex)


def label_final_signs(label: int, nbits: int) -> np.ndarray:
    bits = [(label >> (nbits - 1 - k)) & 1 for k in range(nbits)]
    return np.asarray([1.0 if bit == 0 else -1.0 for bit in bits], dtype=float)


def control_signs_from_final_signs(final_signs: np.ndarray) -> np.ndarray:
    """Return physical reflection signs whose suffix products are final signs.

    The sequential recursion is z_k = s_k z_{k-1} + ...; hence the sign of a
    coefficient inserted at step k is prod_{j=k}^n s_j.  We index the memory
    state by those final suffix signs.  This is an invertible local bit relabeling
    of the physical control qubits, so it does not change the distillable
    entanglement.
    """

    controls = np.empty_like(final_signs)
    if final_signs.size == 0:
        return controls
    controls[:-1] = final_signs[:-1] * final_signs[1:]
    controls[-1] = final_signs[-1]
    return controls


def source_generation_trajectory(
    m: int, spacing: float, eta_source: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return calibrated signal amplitudes and source leakage modes.

    A branch goes through n=log2(M) controlled pi reflections.  At step k:

        drive_k = z_{k-1} + b_k
        reflected_k = s_k drive_k
        environment_k = sqrt(1-eta_s) reflected_k
        z_k = sqrt(eta_s) reflected_k

    The branch-independent displacement/phase choice b_k is chosen so that the
    final retained signal z_n equals the ideal rectangular QAM amplitude.
    """

    if not (0.0 < eta_source <= 1.0):
        raise ValueError("eta_source must satisfy 0 < eta_source <= 1.")

    nbits = int(round(math.log2(m)))
    coeffs = qam_axis_coefficients(m, spacing)
    if coeffs.size != nbits:
        raise ValueError("Coefficient count does not match log2(M).")

    tau = math.sqrt(eta_source)
    leak = math.sqrt(max(0.0, 1.0 - eta_source))
    powers = np.asarray([tau ** (nbits - k) for k in range(nbits)], dtype=float)
    programmed_displacements = coeffs / powers

    final_amps = np.empty(m, dtype=complex)
    source_env = np.empty((m, nbits), dtype=complex)
    max_drive_photons = 0.0

    for label in range(m):
        final_signs = label_final_signs(label, nbits)
        controls = control_signs_from_final_signs(final_signs)
        z = 0.0 + 0.0j
        for k, sign in enumerate(controls):
            drive = z + programmed_displacements[k]
            max_drive_photons = max(max_drive_photons, float(abs(drive) ** 2))
            reflected = sign * drive
            source_env[label, k] = leak * reflected
            z = tau * reflected
        final_amps[label] = z

    target = qam_constellation(m, spacing)
    if not np.allclose(final_amps, target, rtol=2.0e-12, atol=2.0e-12):
        err = float(np.max(np.abs(final_amps - target)))
        raise RuntimeError(f"Source calibration failed; max amplitude error {err:g}.")
    return final_amps, source_env, max_drive_photons


def coherent_overlap_from_modes(branch_modes: np.ndarray) -> np.ndarray:
    """Coherent-state overlap matrix <E_j|E_i> for branch mode vectors."""

    norms = np.sum(np.abs(branch_modes) ** 2, axis=1)
    overlaps = np.exp(
        -0.5 * norms[:, None]
        - 0.5 * norms[None, :]
        + branch_modes @ branch_modes.conj().T
    )
    return (overlaps + overlaps.conj().T) / 2.0


def source_and_channel_loss_coherence(
    final_amps: np.ndarray,
    source_env: np.ndarray,
    eta_channel: float,
) -> np.ndarray:
    channel_env = math.sqrt(max(0.0, 1.0 - eta_channel)) * final_amps[:, None]
    all_env = np.hstack([source_env, channel_env])
    return coherent_overlap_from_modes(all_env)


def evaluate_source_loss_qam_ykl(
    m: int,
    spacing: float,
    eta_source: float,
    eta_channel: float,
    channel_loss_db: float,
    source_loss_db: float,
    rank_tol: float,
) -> SourceLossPoint:
    final_amps, source_env, max_drive_photons = source_generation_trajectory(
        m, spacing, eta_source
    )
    coeffs, _labels, targets = sparse_measurement_coefficients(m, "bell")
    signal_gram, vac_overlaps = coherent_pair_gram_from_amplitudes(
        final_amps, eta_channel
    )
    local_loss = source_and_channel_loss_coherence(
        final_amps, source_env, eta_channel
    )
    std_overlaps, std_gram = standard_overlaps_after_vacuum_subtraction(
        coeffs, signal_gram, vac_overlaps
    )
    ykl_overlaps, ykl_gram = ykl_square_root_measurement(std_overlaps, std_gram)
    result = evaluate_strategy_factorized_loss(
        ykl_overlaps,
        ykl_gram,
        targets,
        local_loss,
        m,
        rank_tol=rank_tol,
    )
    return SourceLossPoint(
        m=m,
        source_loss_db=source_loss_db,
        eta_source=eta_source,
        channel_loss_db=channel_loss_db,
        eta_channel=eta_channel,
        spacing=spacing,
        mean_signal_photon_number=float(np.mean(np.abs(final_amps) ** 2)),
        mean_source_leakage_photons=float(
            np.mean(np.sum(np.abs(source_env) ** 2, axis=1))
        ),
        max_programmed_drive_photons=max_drive_photons,
        result=result,
    )


def _evaluate_task(args: tuple[int, float, float, float, float, float, float]) -> SourceLossPoint:
    m, spacing, source_loss_db, eta_source, channel_loss_db, eta_channel, rank_tol = args
    return evaluate_source_loss_qam_ykl(
        m=m,
        spacing=spacing,
        eta_source=eta_source,
        eta_channel=eta_channel,
        channel_loss_db=channel_loss_db,
        source_loss_db=source_loss_db,
        rank_tol=rank_tol,
    )


def optimize_spacing_for_source_loss(
    m: int,
    source_loss_db: float,
    channel_loss_db: float,
    spacing_min: float,
    spacing_max: float,
    coarse_points: int,
    refine_points: int,
    executor: Executor | None,
    rank_tol: float,
) -> tuple[OptimizedSourceLossRow, list[SourceLossPoint]]:
    eta_source = 10.0 ** (-source_loss_db / 10.0)
    eta_channel = 10.0 ** (-channel_loss_db / 10.0)
    cache: dict[float, SourceLossPoint] = {}

    def evaluate_many(spacings: Iterable[float]) -> None:
        missing = []
        for spacing in spacings:
            key = round(float(spacing), 12)
            if key not in cache:
                missing.append(float(spacing))
        if not missing:
            return
        tasks = [
            (
                m,
                spacing,
                source_loss_db,
                eta_source,
                channel_loss_db,
                eta_channel,
                rank_tol,
            )
            for spacing in missing
        ]
        if executor is None or len(tasks) == 1:
            for task in tasks:
                point = _evaluate_task(task)
                cache[round(point.spacing, 12)] = point
        else:
            for point in executor.map(_evaluate_task, tasks):
                cache[round(point.spacing, 12)] = point

    def get(spacing: float) -> SourceLossPoint:
        key = round(float(spacing), 12)
        if key not in cache:
            evaluate_many([spacing])
        return cache[key]

    coarse = np.linspace(spacing_min, spacing_max, coarse_points)
    evaluate_many(coarse)
    best = max((get(float(spacing)) for spacing in coarse), key=lambda p: p.result.rate)

    step = float(coarse[1] - coarse[0]) if coarse_points > 1 else 0.1
    lo = max(spacing_min, best.spacing - 2.0 * step)
    hi = min(spacing_max, best.spacing + 2.0 * step)
    if math.isclose(best.spacing, spacing_min):
        hi = min(spacing_max, best.spacing + 4.0 * step)
    if math.isclose(best.spacing, spacing_max):
        lo = max(spacing_min, best.spacing - 4.0 * step)

    refine = np.linspace(lo, hi, refine_points)
    evaluate_many(refine)
    best = max([best, *[get(float(spacing)) for spacing in refine]], key=lambda p: p.result.rate)

    row = OptimizedSourceLossRow(
        m=m,
        source_loss_db=source_loss_db,
        eta_source=eta_source,
        channel_loss_db=channel_loss_db,
        eta_channel=eta_channel,
        best_spacing=best.spacing,
        mean_signal_photon_number=best.mean_signal_photon_number,
        mean_source_leakage_photons=best.mean_source_leakage_photons,
        max_programmed_drive_photons=best.max_programmed_drive_photons,
        result=best.result,
    )
    return row, sorted(cache.values(), key=lambda p: p.spacing)


def write_summary_csv(rows: list[OptimizedSourceLossRow], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "M",
                "constellation",
                "qubits_per_side",
                "family",
                "strategy",
                "source_loss_db_per_interaction",
                "eta_source_per_interaction",
                "channel_loss_db_per_arm",
                "eta_channel",
                "best_spacing_d",
                "mean_signal_photon_number",
                "mean_source_leakage_photons",
                "max_programmed_drive_photons",
                "hashing_bound_bits_per_attempt",
                "success_probability",
                "average_target_fidelity",
                "probability_weighted_fidelity",
                "min_coherent_information",
                "useful_outcomes",
            ]
        )
        for row in sorted(rows, key=lambda r: (r.m, r.source_loss_db)):
            writer.writerow(
                [
                    row.m,
                    DISPLAY_NAMES.get(row.m, f"{row.m}-QAM"),
                    int(round(math.log2(row.m))),
                    "bell",
                    "YKL",
                    f"{row.source_loss_db:.12g}",
                    f"{row.eta_source:.12g}",
                    f"{row.channel_loss_db:.12g}",
                    f"{row.eta_channel:.12g}",
                    f"{row.best_spacing:.12g}",
                    f"{row.mean_signal_photon_number:.12g}",
                    f"{row.mean_source_leakage_photons:.12g}",
                    f"{row.max_programmed_drive_photons:.12g}",
                    f"{row.result.rate:.12g}",
                    f"{row.result.success_probability:.12g}",
                    f"{row.result.average_fidelity:.12g}",
                    f"{row.result.success_probability * row.result.average_fidelity:.12g}",
                    f"{row.result.min_coherent_information:.12g}",
                    row.result.useful_outcomes,
                ]
            )


def write_grid_csv(points: list[SourceLossPoint], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "M",
                "constellation",
                "source_loss_db_per_interaction",
                "eta_source_per_interaction",
                "channel_loss_db_per_arm",
                "eta_channel",
                "spacing_d",
                "mean_signal_photon_number",
                "mean_source_leakage_photons",
                "max_programmed_drive_photons",
                "hashing_bound_bits_per_attempt",
                "success_probability",
                "average_target_fidelity",
                "probability_weighted_fidelity",
                "min_coherent_information",
                "useful_outcomes",
            ]
        )
        for point in sorted(points, key=lambda p: (p.m, p.source_loss_db, p.spacing)):
            writer.writerow(
                [
                    point.m,
                    DISPLAY_NAMES.get(point.m, f"{point.m}-QAM"),
                    f"{point.source_loss_db:.12g}",
                    f"{point.eta_source:.12g}",
                    f"{point.channel_loss_db:.12g}",
                    f"{point.eta_channel:.12g}",
                    f"{point.spacing:.12g}",
                    f"{point.mean_signal_photon_number:.12g}",
                    f"{point.mean_source_leakage_photons:.12g}",
                    f"{point.max_programmed_drive_photons:.12g}",
                    f"{point.result.rate:.12g}",
                    f"{point.result.success_probability:.12g}",
                    f"{point.result.average_fidelity:.12g}",
                    f"{point.result.success_probability * point.result.average_fidelity:.12g}",
                    f"{point.result.min_coherent_information:.12g}",
                    point.result.useful_outcomes,
                ]
            )


def write_json(rows: list[OptimizedSourceLossRow], path: Path) -> None:
    payload = []
    for row in sorted(rows, key=lambda r: (r.m, r.source_loss_db)):
        item = asdict(row)
        item["constellation"] = DISPLAY_NAMES.get(row.m, f"{row.m}-QAM")
        item["result"] = asdict(row.result)
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2))


def plot_rate(rows: list[OptimizedSourceLossRow], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.4))
    styles = {
        2: ("#1B6CA8", "o"),
        4: ("#2F9E44", "s"),
        8: ("#D65A31", "^"),
        16: ("#7B2CBF", "D"),
    }
    for m in sorted({row.m for row in rows}):
        subset = sorted([row for row in rows if row.m == m], key=lambda r: r.source_loss_db)
        color, marker = styles.get(m, ("#333333", "o"))
        ax.plot(
            [row.source_loss_db for row in subset],
            [row.result.rate for row in subset],
            color=color,
            marker=marker,
            linewidth=1.9,
            markersize=4.5,
            label=DISPLAY_NAMES.get(m, f"{m}-QAM"),
        )
    ax.set_xlabel("Source-generation loss per qubit interaction (dB)")
    ax.set_ylabel("Optimized hashing bound (bits/attempt)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_spacing(rows: list[OptimizedSourceLossRow], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    styles = {
        2: ("#1B6CA8", "o"),
        4: ("#2F9E44", "s"),
        8: ("#D65A31", "^"),
        16: ("#7B2CBF", "D"),
    }
    for m in sorted({row.m for row in rows}):
        subset = sorted([row for row in rows if row.m == m], key=lambda r: r.source_loss_db)
        color, marker = styles.get(m, ("#333333", "o"))
        ax.plot(
            [row.source_loss_db for row in subset],
            [row.best_spacing for row in subset],
            color=color,
            marker=marker,
            linewidth=1.9,
            markersize=4.5,
            label=DISPLAY_NAMES.get(m, f"{m}-QAM"),
        )
    ax.set_xlabel("Source-generation loss per qubit interaction (dB)")
    ax.set_ylabel("Optimized final QAM spacing d")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_success_and_fidelity(rows: list[OptimizedSourceLossRow], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharex=True)
    styles = {
        2: ("#1B6CA8", "o"),
        4: ("#2F9E44", "s"),
        8: ("#D65A31", "^"),
        16: ("#7B2CBF", "D"),
    }
    for m in sorted({row.m for row in rows}):
        subset = sorted([row for row in rows if row.m == m], key=lambda r: r.source_loss_db)
        color, marker = styles.get(m, ("#333333", "o"))
        x = [row.source_loss_db for row in subset]
        axes[0].plot(
            x,
            [row.result.success_probability for row in subset],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4.0,
            label=DISPLAY_NAMES.get(m, f"{m}-QAM"),
        )
        axes[1].plot(
            x,
            [
                row.result.success_probability * row.result.average_fidelity
                for row in subset
            ],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4.0,
            label=DISPLAY_NAMES.get(m, f"{m}-QAM"),
        )
    axes[0].set_ylabel("Success probability")
    axes[1].set_ylabel("Probability-weighted fidelity")
    for ax in axes:
        ax.set_xlabel("Source-generation loss per interaction (dB)")
        ax.grid(True, alpha=0.25)
    axes[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def source_loss_grid(args: argparse.Namespace) -> list[float]:
    if args.source_losses_db:
        return parse_float_list(args.source_losses_db)
    count = int(round((args.source_loss_max - args.source_loss_min) / args.source_loss_step))
    return [
        round(args.source_loss_min + i * args.source_loss_step, 10)
        for i in range(count + 1)
        if args.source_loss_min + i * args.source_loss_step <= args.source_loss_max + 1.0e-9
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="qam_source_loss_ykl_results")
    parser.add_argument("--m-values", default="2,4,8,16")
    parser.add_argument("--channel-loss-db", type=float, default=0.25)
    parser.add_argument("--source-losses-db", default="")
    parser.add_argument("--source-loss-min", type=float, default=0.0)
    parser.add_argument("--source-loss-max", type=float, default=0.5)
    parser.add_argument("--source-loss-step", type=float, default=0.05)
    parser.add_argument("--spacing-min", type=float, default=0.08)
    parser.add_argument("--spacing-max", type=float, default=4.5)
    parser.add_argument("--coarse-points", type=int, default=25)
    parser.add_argument("--refine-points", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--loss-rank-tol", type=float, default=1.0e-8)
    args = parser.parse_args()

    m_values = parse_int_list(args.m_values)
    for m in m_values:
        qam_shape(m)
    losses = source_loss_grid(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    thread_limiter = None
    try:
        try:
            from threadpoolctl import threadpool_limits

            thread_limiter = threadpool_limits(limits=1)
            thread_limiter.__enter__()
        except Exception:
            thread_limiter = None

        rows: list[OptimizedSourceLossRow] = []
        grid_points: list[SourceLossPoint] = []
        executor_cm = (
            ThreadPoolExecutor(max_workers=args.workers)
            if args.workers and args.workers > 1
            else None
        )
        try:
            if executor_cm is not None:
                executor_cm.__enter__()
            for m in m_values:
                for source_loss_db in losses:
                    print(
                        f"Running {DISPLAY_NAMES.get(m, f'{m}-QAM')} "
                        f"source_loss={source_loss_db:.3g} dB, "
                        f"channel_loss={args.channel_loss_db:.3g} dB"
                    )
                    row, points = optimize_spacing_for_source_loss(
                        m=m,
                        source_loss_db=source_loss_db,
                        channel_loss_db=args.channel_loss_db,
                        spacing_min=args.spacing_min,
                        spacing_max=args.spacing_max,
                        coarse_points=args.coarse_points,
                        refine_points=args.refine_points,
                        executor=executor_cm,
                        rank_tol=args.loss_rank_tol,
                    )
                    rows.append(row)
                    grid_points.extend(points)
                    write_summary_csv(rows, outdir / "source_loss_ykl_summary.csv")
                    write_grid_csv(grid_points, outdir / "source_loss_ykl_spacing_grid.csv")
                    print(
                        f"  R={row.result.rate:.6f}, d={row.best_spacing:.4g}, "
                        f"Psucc={row.result.success_probability:.4f}, "
                        f"Favg={row.result.average_fidelity:.4f}"
                    )
        finally:
            if executor_cm is not None:
                executor_cm.__exit__(None, None, None)

        write_summary_csv(rows, outdir / "source_loss_ykl_summary.csv")
        write_grid_csv(grid_points, outdir / "source_loss_ykl_spacing_grid.csv")
        write_json(rows, outdir / "source_loss_ykl_summary.json")
        plot_rate(rows, outdir / "source_loss_hashing_vs_source_loss.png")
        plot_spacing(rows, outdir / "source_loss_best_spacing_vs_source_loss.png")
        plot_success_and_fidelity(
            rows, outdir / "source_loss_success_weighted_fidelity.png"
        )
        print(f"Wrote results to {outdir}")
    finally:
        if thread_limiter is not None:
            thread_limiter.__exit__(None, None, None)


if __name__ == "__main__":
    main()
