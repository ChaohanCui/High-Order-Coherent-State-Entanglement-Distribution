#!/usr/bin/env python3
"""Reflection-centered QAM source-loss hashing simulations.

This is the compatibility/diagnostic partner to qam_source_loss_hashing.py.
It uses the reflection-centered source trajectories from the earlier
Qubit_Photon_Interface_Loss_scan package, but evaluates them with the current
factorized entropy code and a strict default environment rank cutoff of 1e-8.

The script also supports local relabelings of the generalized Bell/coset
measurement basis.  The physical source labels and their leaked environment
modes are kept fixed; only the Bell-basis convention used by Charlie is
permuted.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
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
    StrategyResult,
    evaluate_strategy_factorized_loss,
    measurement_coefficients,
    sparse_measurement_coefficients,
    ykl_square_root_measurement,
)
from qam_hashing import (
    coherent_pair_gram_from_amplitudes,
    qam_constellation,
    qam_shape,
    standard_overlaps_after_vacuum_subtraction,
)
from compare_schmidt_bell_povm_qam import raw_bell_codeword_overlaps
from qam_source_loss_hashing import coherent_overlap_from_modes


DISPLAY_NAMES = {2: "2-QAM", 4: "4-QAM", 8: "8-QAM", 16: "16-QAM"}


@dataclass(frozen=True)
class ReflectionSource:
    final_amps: np.ndarray
    source_env: np.ndarray
    input_mean_photon_number: float
    n_steps: int


@dataclass(frozen=True)
class EvalPoint:
    m: int
    convention: str
    source_loss_db: float
    eta_source: float
    channel_loss_db: float
    eta_channel: float
    spacing: float
    mean_final_photon_number: float
    input_mean_photon_number: float
    mean_source_leakage_photons: float
    result: StrategyResult


@dataclass(frozen=True)
class OptimizedRow:
    m: int
    convention: str
    source_loss_db: float
    eta_source: float
    channel_loss_db: float
    eta_channel: float
    best_spacing: float
    mean_final_photon_number: float
    input_mean_photon_number: float
    mean_source_leakage_photons: float
    result: StrategyResult


def reflection_centered_source(
    m: int, spacing: float, eta_gen: float, phase_error_rad: float = 0.0
) -> ReflectionSource:
    """Return final amplitudes and leakage modes for the reflection-centered source."""

    d = float(spacing)
    eta = float(eta_gen)
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta_gen must be in (0,1].")
    rt = math.sqrt(eta)
    leak = math.sqrt(max(0.0, 1.0 - eta))
    phase_error = float(phase_error_rad)
    reflected_one = (
        -1.0 + 0.0j
        if phase_error == 0.0
        else np.exp(1j * (math.pi + phase_error))
    )

    def controlled_phase(bit: int) -> complex:
        return 1.0 + 0.0j if bit == 0 else reflected_one

    if m == 2:
        a0 = d / (2.0 * rt)
        final = np.empty(2, dtype=complex)
        eps1 = np.empty(2, dtype=complex)
        for label in range(2):
            x1 = label & 1
            before = controlled_phase(x1) * a0
            eps1[label] = leak * before
            final[label] = rt * before
        return ReflectionSource(final, np.vstack([eps1]).T, a0 * a0, 1)

    if m == 4:
        a0 = d / (2.0 * eta)
        a = rt * a0
        u = math.sqrt(2.0) * a
        radius = d / math.sqrt(2.0)
        rot = np.exp(-1j * math.pi / 4.0)
        final_diamond = {
            (0, 0): +radius,
            (0, 1): -radius,
            (1, 0): -1j * radius,
            (1, 1): +1j * radius,
        }
        before_second_base = {0: +u, 1: -1j * u}
        final = np.empty(4, dtype=complex)
        eps1 = np.empty(4, dtype=complex)
        eps2 = np.empty(4, dtype=complex)
        for label in range(4):
            x1 = (label >> 1) & 1
            x2 = label & 1
            eps1[label] = leak * controlled_phase(x1) * a0
            before2 = controlled_phase(x2) * before_second_base[x1]
            eps2[label] = leak * before2
            if phase_error_rad == 0.0:
                final[label] = rot * final_diamond[(x1, x2)]
            else:
                final[label] = rot * rt * before2
        return ReflectionSource(final, np.vstack([eps1, eps2]).T, a0 * a0, 2)

    if m == 8:
        a0 = d / (2.0 * (eta**1.5))
        a = rt * a0
        u = math.sqrt(2.0) * a
        scale = d / rt
        before_second_base = {0: +u, 1: -1j * u}
        l3 = {
            (0, 0): -0.5 * scale - 0.5j * scale,
            (0, 1): -1.5 * scale + 0.5j * scale,
            (1, 0): -1.5 * scale - 0.5j * scale,
            (1, 1): -0.5 * scale + 0.5j * scale,
        }
        final = np.empty(8, dtype=complex)
        eps1 = np.empty(8, dtype=complex)
        eps2 = np.empty(8, dtype=complex)
        eps3 = np.empty(8, dtype=complex)
        for label in range(8):
            x1 = (label >> 2) & 1
            x2 = (label >> 1) & 1
            x3 = label & 1
            eps1[label] = leak * controlled_phase(x1) * a0
            eps2[label] = leak * controlled_phase(x2) * before_second_base[x1]
            before3 = controlled_phase(x3) * l3[(x1, x2)]
            eps3[label] = leak * before3
            final[label] = rt * before3
        return ReflectionSource(final, np.vstack([eps1, eps2, eps3]).T, a0 * a0, 3)

    if m == 16:
        d2 = d / (eta**1.5)
        a0 = d2 / (2.0 * rt)
        d3 = d / eta
        d4 = d / rt
        final = np.empty(16, dtype=complex)
        eps = [np.empty(16, dtype=complex) for _ in range(4)]
        for label in range(16):
            x1 = (label >> 3) & 1
            x2 = (label >> 2) & 1
            x3 = (label >> 1) & 1
            x4 = label & 1
            before1 = controlled_phase(x1) * a0
            eps[0][label] = leak * before1
            z1 = rt * before1
            before2 = controlled_phase(x2) * z1 + 2.0 * d2 * x2 - d2
            eps[1][label] = leak * before2
            z2 = rt * before2
            before3 = controlled_phase(x3) * z2 + 1j * d3 * x3 - 0.5j * d3
            eps[2][label] = leak * before3
            z3 = rt * before3
            before4 = controlled_phase(x4) * z3 + 2j * d4 * x4 - 1j * d4
            eps[3][label] = leak * before4
            final[label] = rt * before4
        return ReflectionSource(final, np.vstack(eps).T, a0 * a0, 4)

    raise ValueError("Reflection-centered source supports M=2,4,8,16.")


def source_and_channel_loss_coherence(
    final_amps: np.ndarray, source_env: np.ndarray, eta_channel: float
) -> np.ndarray:
    channel_env = math.sqrt(max(0.0, 1.0 - eta_channel)) * final_amps[:, None]
    return coherent_overlap_from_modes(np.hstack([source_env, channel_env]))


def normalized_rows(rows: np.ndarray, eps: float = EPS) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1)
    out = np.zeros_like(rows)
    keep = norms > eps
    out[keep] = rows[keep] / norms[keep, None]
    return out


def nearest_constellation_permutation(final_amps: np.ndarray, spacing: float) -> np.ndarray:
    """physical_for_logical permutation matching qam_constellation labels."""

    m = final_amps.size
    target = qam_constellation(m, spacing)
    used: set[int] = set()
    perm = np.empty(m, dtype=int)
    for logical, amp in enumerate(target):
        distances = np.abs(final_amps - amp)
        order = np.argsort(distances)
        physical = next(int(idx) for idx in order if int(idx) not in used)
        if distances[physical] > 1.0e-8 * max(1.0, abs(spacing)):
            raise RuntimeError(
                f"Could not match logical label {logical}; nearest error={distances[physical]:g}"
            )
        perm[logical] = physical
        used.add(physical)
    return perm


def bit_permutation(m: int, order: tuple[int, ...], mask: int = 0) -> np.ndarray:
    nbits = int(round(math.log2(m)))
    perm = np.empty(m, dtype=int)
    for logical in range(m):
        bits = [(logical >> (nbits - 1 - k)) & 1 for k in range(nbits)]
        outbits = [bits[order[k]] for k in range(nbits)]
        out = 0
        for bit in outbits:
            out = (out << 1) | bit
        perm[logical] = out ^ mask
    return perm


def convention_permutation(
    convention: str, m: int, final_amps: np.ndarray, spacing: float
) -> np.ndarray:
    nbits = int(round(math.log2(m)))
    if convention == "reflection":
        return np.arange(m, dtype=int)
    if convention == "natural":
        return nearest_constellation_permutation(final_amps, spacing)
    if convention == "bit_reverse":
        return bit_permutation(m, tuple(reversed(range(nbits))))
    if convention.startswith("bitperm:"):
        spec = convention.split(":", 1)[1]
        order_text, _, mask_text = spec.partition("^")
        order = tuple(int(ch) for ch in order_text)
        mask = int(mask_text) if mask_text else 0
        return bit_permutation(m, order, mask)
    raise ValueError(f"Unknown convention: {convention}")


def permute_bell_coefficients(
    m: int, physical_for_logical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coeffs, _labels, targets = measurement_coefficients(m, "bell")
    pair_perm = np.empty(m * m, dtype=int)
    for a in range(m):
        for b in range(m):
            pair_perm[a * m + b] = physical_for_logical[a] * m + physical_for_logical[b]
    coeffs_phys = np.zeros_like(coeffs)
    targets_phys = np.zeros_like(targets)
    coeffs_phys[:, pair_perm] = coeffs
    targets_phys[:, pair_perm] = targets
    return coeffs_phys, targets_phys


def evaluate_reflection_srm(
    m: int,
    spacing: float,
    eta_source: float,
    eta_channel: float,
    source_loss_db: float,
    channel_loss_db: float,
    convention: str,
    rank_tol: float,
    phase_error_rad: float = 0.0,
    receiver: str = "vacuum_omit_srm",
) -> EvalPoint:
    source = reflection_centered_source(
        m, spacing, eta_source, phase_error_rad=phase_error_rad
    )
    if convention == "reflection":
        coeffs, _labels, targets = sparse_measurement_coefficients(m, "bell")
    else:
        perm = convention_permutation(convention, m, source.final_amps, spacing)
        coeffs, targets = permute_bell_coefficients(m, perm)
    gram, vac_overlaps = coherent_pair_gram_from_amplitudes(
        source.final_amps, eta_channel
    )
    if receiver == "vacuum_omit_srm":
        std_overlaps, std_gram = standard_overlaps_after_vacuum_subtraction(
            coeffs, gram, vac_overlaps
        )
        eval_targets = targets
    elif receiver == "raw_label_srm":
        std_overlaps, std_gram = raw_bell_codeword_overlaps(coeffs, gram)
        eval_targets = None
    else:
        raise ValueError(f"Unknown receiver: {receiver}")
    ykl_overlaps, ykl_gram = ykl_square_root_measurement(std_overlaps, std_gram)
    if eval_targets is None:
        eval_targets = normalized_rows(ykl_overlaps)
    local_loss = source_and_channel_loss_coherence(
        source.final_amps, source.source_env, eta_channel
    )
    result = evaluate_strategy_factorized_loss(
        ykl_overlaps, ykl_gram, eval_targets, local_loss, m, rank_tol=rank_tol
    )
    return EvalPoint(
        m=m,
        convention=convention,
        source_loss_db=source_loss_db,
        eta_source=eta_source,
        channel_loss_db=channel_loss_db,
        eta_channel=eta_channel,
        spacing=spacing,
        mean_final_photon_number=float(np.mean(np.abs(source.final_amps) ** 2)),
        input_mean_photon_number=source.input_mean_photon_number,
        mean_source_leakage_photons=float(
            np.mean(np.sum(np.abs(source.source_env) ** 2, axis=1))
        ),
        result=result,
    )


def evaluate_reflection_ykl(
    m: int,
    spacing: float,
    eta_source: float,
    eta_channel: float,
    source_loss_db: float,
    channel_loss_db: float,
    convention: str,
    rank_tol: float,
    phase_error_rad: float = 0.0,
) -> EvalPoint:
    return evaluate_reflection_srm(
        m=m,
        spacing=spacing,
        eta_source=eta_source,
        eta_channel=eta_channel,
        source_loss_db=source_loss_db,
        channel_loss_db=channel_loss_db,
        convention=convention,
        rank_tol=rank_tol,
        phase_error_rad=phase_error_rad,
        receiver="vacuum_omit_srm",
    )


def _eval_task(args: tuple[int, float, float, float, float, float, str, float]) -> EvalPoint:
    m, spacing, eta_source, eta_channel, source_loss_db, channel_loss_db, convention, rank_tol = args
    return evaluate_reflection_ykl(
        m=m,
        spacing=spacing,
        eta_source=eta_source,
        eta_channel=eta_channel,
        source_loss_db=source_loss_db,
        channel_loss_db=channel_loss_db,
        convention=convention,
        rank_tol=rank_tol,
    )


def optimize_spacing(
    m: int,
    source_loss_db: float,
    channel_loss_db: float,
    convention: str,
    spacing_min: float,
    spacing_max: float,
    coarse_points: int,
    refine_points: int,
    executor: Executor | None,
    rank_tol: float,
) -> tuple[OptimizedRow, list[EvalPoint]]:
    eta_source = 10.0 ** (-source_loss_db / 10.0)
    eta_channel = 10.0 ** (-channel_loss_db / 10.0)
    cache: dict[float, EvalPoint] = {}

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
                eta_source,
                eta_channel,
                source_loss_db,
                channel_loss_db,
                convention,
                rank_tol,
            )
            for spacing in missing
        ]
        if executor is None or len(tasks) == 1:
            for task in tasks:
                point = _eval_task(task)
                cache[round(point.spacing, 12)] = point
        else:
            for point in executor.map(_eval_task, tasks):
                cache[round(point.spacing, 12)] = point

    def get(spacing: float) -> EvalPoint:
        key = round(float(spacing), 12)
        if key not in cache:
            evaluate_many([spacing])
        return cache[key]

    coarse = np.linspace(spacing_min, spacing_max, coarse_points)
    evaluate_many(coarse)
    best = max((get(float(d)) for d in coarse), key=lambda p: p.result.rate)
    step = float(coarse[1] - coarse[0]) if coarse_points > 1 else 0.1
    lo = max(spacing_min, best.spacing - 2.0 * step)
    hi = min(spacing_max, best.spacing + 2.0 * step)
    if math.isclose(best.spacing, spacing_min):
        hi = min(spacing_max, best.spacing + 4.0 * step)
    if math.isclose(best.spacing, spacing_max):
        lo = max(spacing_min, best.spacing - 4.0 * step)
    refine = np.linspace(lo, hi, refine_points)
    evaluate_many(refine)
    best = max([best, *[get(float(d)) for d in refine]], key=lambda p: p.result.rate)
    row = OptimizedRow(
        m=m,
        convention=convention,
        source_loss_db=source_loss_db,
        eta_source=eta_source,
        channel_loss_db=channel_loss_db,
        eta_channel=eta_channel,
        best_spacing=best.spacing,
        mean_final_photon_number=best.mean_final_photon_number,
        input_mean_photon_number=best.input_mean_photon_number,
        mean_source_leakage_photons=best.mean_source_leakage_photons,
        result=best.result,
    )
    return row, sorted(cache.values(), key=lambda p: p.spacing)


def write_summary_csv(rows: list[OptimizedRow], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "M",
                "constellation",
                "measurement_convention",
                "channel_loss_db",
                "eta_channel",
                "generation_loss_db_per_step",
                "eta_gen_each_step",
                "qubits_per_node",
                "optimized_spacing_d",
                "mean_final_photon_number",
                "input_mean_photon_number",
                "mean_source_leakage_photons",
                "strategy",
                "rank_tol",
                "hashing_bound_bits_per_attempt",
                "success_probability",
                "average_target_fidelity",
                "probability_weighted_fidelity",
                "min_coherent_information",
                "useful_outcomes",
            ]
        )
        for row in sorted(rows, key=lambda r: (r.convention, r.m, r.source_loss_db)):
            writer.writerow(
                [
                    row.m,
                    DISPLAY_NAMES.get(row.m, f"{row.m}-QAM"),
                    row.convention,
                    f"{row.channel_loss_db:.12g}",
                    f"{row.eta_channel:.16g}",
                    f"{row.source_loss_db:.12g}",
                    f"{row.eta_source:.16g}",
                    int(round(math.log2(row.m))),
                    f"{row.best_spacing:.16g}",
                    f"{row.mean_final_photon_number:.16g}",
                    f"{row.input_mean_photon_number:.16g}",
                    f"{row.mean_source_leakage_photons:.16g}",
                    "YKL",
                    "1e-8",
                    f"{row.result.rate:.16g}",
                    f"{row.result.success_probability:.16g}",
                    f"{row.result.average_fidelity:.16g}",
                    f"{row.result.success_probability * row.result.average_fidelity:.16g}",
                    f"{row.result.min_coherent_information:.16g}",
                    row.result.useful_outcomes,
                ]
            )


def write_grid_csv(points: list[EvalPoint], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "M",
                "constellation",
                "measurement_convention",
                "channel_loss_db",
                "eta_channel",
                "generation_loss_db_per_step",
                "eta_gen_each_step",
                "spacing_d",
                "hashing_bound_bits_per_attempt",
                "success_probability",
                "average_target_fidelity",
                "probability_weighted_fidelity",
                "min_coherent_information",
                "useful_outcomes",
            ]
        )
        for p in sorted(points, key=lambda r: (r.convention, r.m, r.source_loss_db, r.spacing)):
            writer.writerow(
                [
                    p.m,
                    DISPLAY_NAMES.get(p.m, f"{p.m}-QAM"),
                    p.convention,
                    f"{p.channel_loss_db:.12g}",
                    f"{p.eta_channel:.16g}",
                    f"{p.source_loss_db:.12g}",
                    f"{p.eta_source:.16g}",
                    f"{p.spacing:.16g}",
                    f"{p.result.rate:.16g}",
                    f"{p.result.success_probability:.16g}",
                    f"{p.result.average_fidelity:.16g}",
                    f"{p.result.success_probability * p.result.average_fidelity:.16g}",
                    f"{p.result.min_coherent_information:.16g}",
                    p.result.useful_outcomes,
                ]
            )


def write_json(rows: list[OptimizedRow], path: Path) -> None:
    payload = []
    for row in sorted(rows, key=lambda r: (r.convention, r.m, r.source_loss_db)):
        item = asdict(row)
        item["constellation"] = DISPLAY_NAMES.get(row.m, f"{row.m}-QAM")
        item["result"] = asdict(row.result)
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2))


def plot_rates(rows: list[OptimizedRow], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = {2: "#1B6CA8", 4: "#2F9E44", 8: "#D65A31", 16: "#7B2CBF"}
    markers = {"reflection": "o", "natural": "s"}
    for convention in sorted({r.convention for r in rows}):
        for m in sorted({r.m for r in rows if r.convention == convention}):
            subset = sorted(
                [r for r in rows if r.m == m and r.convention == convention],
                key=lambda r: r.source_loss_db,
            )
            ax.plot(
                [r.source_loss_db for r in subset],
                [r.result.rate for r in subset],
                color=colors.get(m, "#333333"),
                marker=markers.get(convention, "x"),
                linestyle="-" if convention == "reflection" else "--",
                linewidth=1.8,
                markersize=4.2,
                label=f"{DISPLAY_NAMES.get(m, f'{m}-QAM')} {convention}",
            )
    ax.set_xlabel("Generation loss per qubit interaction (dB)")
    ax.set_ylabel("Optimized hashing bound (bits/attempt)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_spacing(rows: list[OptimizedRow], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {2: "#1B6CA8", 4: "#2F9E44", 8: "#D65A31", 16: "#7B2CBF"}
    for m in sorted({r.m for r in rows}):
        subset = sorted(
            [r for r in rows if r.m == m and r.convention == "reflection"],
            key=lambda r: r.source_loss_db,
        )
        if not subset:
            continue
        ax.plot(
            [r.source_loss_db for r in subset],
            [r.best_spacing for r in subset],
            color=colors.get(m, "#333333"),
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            label=DISPLAY_NAMES.get(m, f"{m}-QAM"),
        )
    ax.set_xlabel("Generation loss per qubit interaction (dB)")
    ax.set_ylabel("Optimized spacing d, reflection convention")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def loss_grid(args: argparse.Namespace) -> list[float]:
    if args.source_losses_db:
        return parse_float_list(args.source_losses_db)
    count = int(round((args.source_loss_max - args.source_loss_min) / args.source_loss_step))
    return [
        round(args.source_loss_min + idx * args.source_loss_step, 10)
        for idx in range(count + 1)
        if args.source_loss_min + idx * args.source_loss_step <= args.source_loss_max + 1.0e-9
    ]


def bitperm_conventions(m: int, include_masks: bool) -> list[str]:
    nbits = int(round(math.log2(m)))
    out = []
    masks = range(m) if include_masks else [0]
    for order in itertools.permutations(range(nbits)):
        order_text = "".join(str(x) for x in order)
        for mask in masks:
            out.append(f"bitperm:{order_text}^{mask}")
    return out


def run_label_search(args: argparse.Namespace, outdir: Path) -> None:
    search_m = parse_int_list(args.search_m_values)
    search_losses = parse_float_list(args.search_losses_db)
    rows: list[EvalPoint] = []
    for m in search_m:
        conventions = ["reflection", "natural", "bit_reverse"]
        conventions.extend(bitperm_conventions(m, include_masks=args.search_include_masks))
        # Preserve order but remove duplicates.
        conventions = list(dict.fromkeys(conventions))
        eta_channel = 10.0 ** (-args.channel_loss_db / 10.0)
        for source_loss_db in search_losses:
            eta_source = 10.0 ** (-source_loss_db / 10.0)
            source = reflection_centered_source(m, args.search_spacing, eta_source)
            for convention in conventions:
                point = evaluate_reflection_ykl(
                    m=m,
                    spacing=args.search_spacing,
                    eta_source=eta_source,
                    eta_channel=eta_channel,
                    source_loss_db=source_loss_db,
                    channel_loss_db=args.channel_loss_db,
                    convention=convention,
                    rank_tol=args.loss_rank_tol,
                )
                rows.append(point)
            best = max(
                [r for r in rows if r.m == m and math.isclose(r.source_loss_db, source_loss_db)],
                key=lambda r: r.result.rate,
            )
            print(
                f"label search M={m}, loss={source_loss_db:g}, d={args.search_spacing:g}: "
                f"best={best.convention}, R={best.result.rate:.6g}"
            )

    path = outdir / "reflection_label_convention_search.csv"
    write_grid_csv(rows, path)
    top_path = outdir / "reflection_label_convention_search_top.csv"
    with top_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "M",
                "generation_loss_db_per_step",
                "spacing_d",
                "rank",
                "measurement_convention",
                "hashing_bound_bits_per_attempt",
                "success_probability",
                "average_target_fidelity",
                "min_coherent_information",
            ]
        )
        for m in search_m:
            for source_loss_db in search_losses:
                subset = sorted(
                    [
                        r
                        for r in rows
                        if r.m == m and math.isclose(r.source_loss_db, source_loss_db)
                    ],
                    key=lambda r: r.result.rate,
                    reverse=True,
                )
                for rank, r in enumerate(subset[: min(12, len(subset))], start=1):
                    writer.writerow(
                        [
                            r.m,
                            f"{r.source_loss_db:.12g}",
                            f"{r.spacing:.12g}",
                            rank,
                            r.convention,
                            f"{r.result.rate:.16g}",
                            f"{r.result.success_probability:.16g}",
                            f"{r.result.average_fidelity:.16g}",
                            f"{r.result.min_coherent_information:.16g}",
                        ]
                    )
    print(f"Wrote {path}")
    print(f"Wrote {top_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="qam_reflection_source_loss_ykl_results")
    parser.add_argument("--m-values", default="2,4,8,16")
    parser.add_argument("--conventions", default="reflection")
    parser.add_argument("--channel-loss-db", type=float, default=0.25)
    parser.add_argument("--source-losses-db", default="")
    parser.add_argument("--source-loss-min", type=float, default=0.0)
    parser.add_argument("--source-loss-max", type=float, default=0.5)
    parser.add_argument("--source-loss-step", type=float, default=0.05)
    parser.add_argument("--spacing-min", type=float, default=0.1)
    parser.add_argument("--spacing-max", type=float, default=6.0)
    parser.add_argument("--coarse-points", type=int, default=21)
    parser.add_argument("--refine-points", type=int, default=21)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--loss-rank-tol", type=float, default=1.0e-8)
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--search-label-conventions", action="store_true")
    parser.add_argument("--search-m-values", default="8,16")
    parser.add_argument("--search-losses-db", default="0.1,0.25,0.5")
    parser.add_argument("--search-spacing", type=float, default=1.0)
    parser.add_argument("--search-include-masks", action="store_true")
    args = parser.parse_args()

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

        rows: list[OptimizedRow] = []
        grid_points: list[EvalPoint] = []
        if not args.skip_scan:
            m_values = parse_int_list(args.m_values)
            conventions = [x.strip() for x in args.conventions.split(",") if x.strip()]
            losses = loss_grid(args)
            executor_cm = (
                ThreadPoolExecutor(max_workers=args.workers)
                if args.workers and args.workers > 1
                else None
            )
            try:
                if executor_cm is not None:
                    executor_cm.__enter__()
                for convention in conventions:
                    for m in m_values:
                        qam_shape(m)
                        for source_loss_db in losses:
                            print(
                                f"Running {DISPLAY_NAMES.get(m, f'{m}-QAM')} "
                                f"conv={convention}, gen_loss={source_loss_db:.3g} dB"
                            )
                            row, points = optimize_spacing(
                                m=m,
                                source_loss_db=source_loss_db,
                                channel_loss_db=args.channel_loss_db,
                                convention=convention,
                                spacing_min=args.spacing_min,
                                spacing_max=args.spacing_max,
                                coarse_points=args.coarse_points,
                                refine_points=args.refine_points,
                                executor=executor_cm,
                                rank_tol=args.loss_rank_tol,
                            )
                            rows.append(row)
                            grid_points.extend(points)
                            write_summary_csv(rows, outdir / "reflection_source_ykl_summary.csv")
                            write_grid_csv(grid_points, outdir / "reflection_source_ykl_spacing_grid.csv")
                            print(
                                f"  R={row.result.rate:.6g}, d={row.best_spacing:.6g}, "
                                f"P={row.result.success_probability:.6g}, "
                                f"F={row.result.average_fidelity:.6g}"
                            )
            finally:
                if executor_cm is not None:
                    executor_cm.__exit__(None, None, None)

            write_summary_csv(rows, outdir / "reflection_source_ykl_summary.csv")
            write_grid_csv(grid_points, outdir / "reflection_source_ykl_spacing_grid.csv")
            write_json(rows, outdir / "reflection_source_ykl_summary.json")
            plot_rates(rows, outdir / "reflection_source_hashing_vs_generation_loss.png")
            plot_spacing(rows, outdir / "reflection_source_best_spacing_vs_generation_loss.png")

        if args.search_label_conventions:
            run_label_search(args, outdir)

        metadata = {
            "channel_loss_db": args.channel_loss_db,
            "loss_rank_tol": args.loss_rank_tol,
            "source_model": "reflection-centered",
            "m_values": args.m_values,
            "conventions": args.conventions,
            "source_losses_db": args.source_losses_db or {
                "min": args.source_loss_min,
                "max": args.source_loss_max,
                "step": args.source_loss_step,
            },
            "coarse_points": args.coarse_points,
            "refine_points": args.refine_points,
        }
        (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"Wrote results to {outdir}")
    finally:
        if thread_limiter is not None:
            thread_limiter.__exit__(None, None, None)


if __name__ == "__main__":
    main()
