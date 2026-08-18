#!/usr/bin/env python3
"""Coherent-state assisted M-PSK entanglement distribution sweeps.

This is a compact, analytic version of the notebook machinery for the
GHZ-pair and full Bell/coset measurement families.  It avoids a Fock-space
truncation by using exact coherent-state overlaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse


EPS = 1.0e-12
LOSS_RANK_TOL = 1.0e-12


@dataclass(frozen=True)
class StrategyResult:
    rate: float
    success_probability: float
    average_fidelity: float
    min_coherent_information: float
    useful_outcomes: int


@dataclass(frozen=True)
class SweepBest:
    family: str
    m: int
    loss_db: float
    eta: float
    strategy: str
    best_alpha: float
    result: StrategyResult


def entropy_bits(rho: np.ndarray) -> float:
    vals = scipy.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    vals = np.real(vals)
    vals[abs(vals) < 1.0e-13] = 0.0
    vals = np.clip(vals, 0.0, None)
    total = np.sum(vals)
    if total <= EPS:
        return 0.0
    vals = vals / total
    nz = vals[vals > 1.0e-14]
    return float(-np.sum(nz * np.log2(nz)))


def entropy_bits_from_eigvals(evals: np.ndarray) -> np.ndarray:
    vals = np.real(evals).copy()
    vals[np.abs(vals) < 1.0e-13] = 0.0
    vals = np.clip(vals, 0.0, None)
    totals = np.sum(vals, axis=-1, keepdims=True)
    safe = totals[..., 0] > EPS
    out = np.zeros(vals.shape[:-1], dtype=float)
    if np.any(safe):
        probs = vals[safe] / totals[safe]
        logs = np.zeros_like(probs)
        np.log2(probs, out=logs, where=probs > 1.0e-14)
        out[safe] = -np.sum(np.where(probs > 1.0e-14, probs * logs, 0.0), axis=-1)
    return out


@lru_cache(maxsize=None)
def pair_phase_arrays(m: int) -> tuple[np.ndarray, np.ndarray]:
    phases = np.exp(1j * 2.0 * np.pi * np.arange(m) / m)
    p_a = np.empty(m * m, dtype=complex)
    p_b = np.empty(m * m, dtype=complex)
    for a in range(m):
        for b in range(m):
            idx = a * m + b
            p_a[idx] = phases[a]
            p_b[idx] = phases[b]
    return p_a, p_b


def coherent_pair_gram(alpha: float, eta: float, m: int) -> np.ndarray:
    p_a, p_b = pair_phase_arrays(m)
    beta2 = eta * alpha * alpha
    gram = np.exp(
        -beta2
        * (
            2.0
            - np.outer(np.conj(p_a), p_a)
            - np.outer(np.conj(p_b), p_b)
        )
    )
    return (gram + gram.conj().T) / 2.0


def loss_coherence_matrix(alpha: float, eta: float, m: int) -> np.ndarray:
    p_a, p_b = pair_phase_arrays(m)
    env2 = (1.0 - eta) * alpha * alpha
    loss = np.exp(
        -env2
        * (
            2.0
            - np.outer(p_a, np.conj(p_a))
            - np.outer(p_b, np.conj(p_b))
        )
    )
    return loss


def local_loss_coherence_matrix(alpha: float, eta: float, m: int) -> np.ndarray:
    phases = np.exp(1j * 2.0 * np.pi * np.arange(m) / m)
    env2 = (1.0 - eta) * alpha * alpha
    loss = np.exp(-env2 * (1.0 - np.outer(phases, np.conj(phases))))
    # The expression above is <e_j|e_i> for a single arm.  Symmetrize to
    # remove tiny floating-point skew before eigensolvers see it.
    return (loss + loss.conj().T) / 2.0


def local_loss_coherence_from_amplitudes(local_amps: np.ndarray, eta: float) -> np.ndarray:
    env_amps = np.sqrt(1.0 - eta) * local_amps
    norms = np.abs(env_amps) ** 2
    loss = np.exp(
        -0.5 * norms[:, None]
        - 0.5 * norms[None, :]
        + np.outer(env_amps, np.conj(env_amps))
    )
    return (loss + loss.conj().T) / 2.0


def bits(x: int, n: int) -> str:
    return format(x, f"0{n}b")


def ghz_pair_coefficients(m: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return two-branch GHZ-pair optical/memory target coefficients.

    Each computational pair (a,b) is paired with its bitwise complement
    (a xor (m-1), b xor (m-1)).  The +/- superpositions are locally equivalent
    GHZ states across the 2 log2(m) memory qubits.
    """

    dim = m * m
    mask = m - 1
    nbits = int(round(math.log2(m)))
    visited: set[int] = set()
    coeffs: list[np.ndarray] = []
    labels: list[str] = []
    targets: list[np.ndarray] = []

    for a in range(m):
        for b in range(m):
            idx = a * m + b
            cidx = (a ^ mask) * m + (b ^ mask)
            if idx in visited or cidx in visited:
                continue
            visited.add(idx)
            visited.add(cidx)
            for sign, sign_label in [(1.0, "+"), (-1.0, "-")]:
                c = np.zeros(dim, dtype=complex)
                c[idx] = 1.0
                c[cidx] = sign
                coeffs.append(c)

                target = np.zeros(dim, dtype=complex)
                target[idx] = 1.0 / math.sqrt(2.0)
                target[cidx] = sign / math.sqrt(2.0)
                targets.append(target)

                labels.append(
                    f"|{bits(a, nbits)}{bits(b, nbits)}> {sign_label} "
                    f"|{bits(a ^ mask, nbits)}{bits(b ^ mask, nbits)}>"
                )

    return np.vstack(coeffs), labels, np.vstack(targets)


def bell_coset_coefficients(m: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return M-branch maximally entangled qudit/qubit-register coefficients.

    The basis is the finite-group Bell basis over q local qubits:

        |B_{r,s}> = 1/sqrt(M) sum_h (-1)^{s.h} |h>_A |h xor r>_B .

    These states have Schmidt rank M and hence log2(M) ebits at zero loss.
    """

    dim = m * m
    nbits = int(round(math.log2(m)))
    coeffs: list[np.ndarray] = []
    labels: list[str] = []
    targets: list[np.ndarray] = []

    for r in range(m):
        for s in range(m):
            c = np.zeros(dim, dtype=complex)
            terms: list[str] = []
            for h in range(m):
                parity = (s & h).bit_count() % 2
                sign = -1.0 if parity else 1.0
                idx = h * m + (h ^ r)
                c[idx] = sign
                if h < 4:
                    prefix = "-" if sign < 0 else "+"
                    terms.append(f"{prefix}|{bits(h, nbits)}{bits(h ^ r, nbits)}>")
            coeffs.append(c)
            targets.append(c / math.sqrt(m))
            labels.append(" ".join(terms))

    return np.vstack(coeffs), labels, np.vstack(targets)


@lru_cache(maxsize=None)
def measurement_coefficients(
    m: int, family: str
) -> tuple[np.ndarray, list[str], np.ndarray]:
    if family == "ghz":
        return ghz_pair_coefficients(m)
    if family == "bell":
        return bell_coset_coefficients(m)
    raise ValueError(f"Unknown measurement family: {family}")


@lru_cache(maxsize=None)
def sparse_measurement_coefficients(
    m: int, family: str
) -> tuple[scipy.sparse.csr_matrix, list[str], np.ndarray]:
    coeffs, labels, targets = measurement_coefficients(m, family)
    return scipy.sparse.csr_matrix(coeffs), labels, targets


def standard_overlaps_after_vacuum_subtraction(
    coeffs: np.ndarray, gram: np.ndarray, alpha: float, eta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute <phi_k|optical_j> and <phi_k|phi_l> for standard vectors."""

    dim = gram.shape[0]
    v0 = np.full(dim, np.exp(-eta * alpha * alpha), dtype=complex)
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
    vac_overlap = np.asarray(coeffs @ v0).ravel() / raw_norm
    residual_norm2 = np.maximum(1.0 - np.abs(vac_overlap) ** 2, EPS)
    residual_norm = np.sqrt(residual_norm2)

    overlaps = (
        psi_overlaps - np.conj(vac_overlap)[:, None] * v0[None, :]
    ) / residual_norm[:, None]

    psi_gram = raw_gram / (raw_norm[:, None] * raw_norm[None, :])
    vec_gram = (
        psi_gram - np.conj(vac_overlap)[:, None] * vac_overlap[None, :]
    ) / (residual_norm[:, None] * residual_norm[None, :])
    vec_gram = (vec_gram + vec_gram.conj().T) / 2.0

    return overlaps, vec_gram


def ykl_square_root_measurement(
    standard_overlaps: np.ndarray, standard_gram: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = scipy.linalg.eigh((standard_gram + standard_gram.conj().T) / 2.0)
    inv_sqrt = np.zeros_like(evals)
    mask = evals > 1.0e-10
    inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    w = (evecs * inv_sqrt) @ evecs.conj().T
    ykl_overlaps = w.conj().T @ standard_overlaps
    ykl_gram = w.conj().T @ standard_gram @ w
    ykl_gram = (ykl_gram + ykl_gram.conj().T) / 2.0
    return ykl_overlaps, ykl_gram


def povm_scale(vec_gram: np.ndarray) -> float:
    evals = scipy.linalg.eigvalsh((vec_gram + vec_gram.conj().T) / 2.0)
    max_eval = float(np.max(np.real(evals)))
    if max_eval <= 1.0:
        return 1.0
    return 1.0 / max_eval


def evaluate_strategy(
    overlaps: np.ndarray,
    vec_gram: np.ndarray,
    targets: np.ndarray,
    loss_matrix: np.ndarray,
    m: int,
) -> StrategyResult:
    dim = m * m
    scale = povm_scale(vec_gram)
    total_rate = 0.0
    total_success = 0.0
    fidelity_weight = 0.0
    min_ci = float("inf")
    useful_outcomes = 0

    n_outcomes = overlaps.shape[0]
    # Keep chunk memory below about 128 MiB for the M=16 case.
    bytes_per_rho = dim * dim * np.dtype(np.complex128).itemsize
    chunk_size = max(1, min(n_outcomes, int((128 * 1024 * 1024) / bytes_per_rho)))

    for start in range(0, n_outcomes, chunk_size):
        stop = min(start + chunk_size, n_outcomes)
        v = overlaps[start:stop]
        target = targets[start:stop]

        raw_probs = np.sum(np.abs(v) ** 2, axis=1).real / dim
        valid = raw_probs > 1.0e-14
        if not np.any(valid):
            continue

        v = v[valid]
        target = target[valid]
        raw_probs = raw_probs[valid]
        rho_raw = (
            loss_matrix[None, :, :]
            * v[:, :, None]
            * np.conj(v[:, None, :])
            / dim
        )
        rho_raw = (rho_raw + np.conj(np.swapaxes(rho_raw, -1, -2))) / 2.0
        rho = rho_raw / raw_probs[:, None, None]

        evals_ab = np.linalg.eigvalsh(rho)
        s_ab = entropy_bits_from_eigvals(evals_ab)

        rho_a = np.einsum("kabcb->kac", rho.reshape(-1, m, m, m, m))
        rho_a = (rho_a + np.conj(np.swapaxes(rho_a, -1, -2))) / 2.0
        evals_a = np.linalg.eigvalsh(rho_a)
        s_a = entropy_bits_from_eigvals(evals_a)

        coherent_information = s_a - s_ab
        probs = scale * raw_probs
        useful = np.maximum(coherent_information, 0.0)

        total_rate += float(np.sum(probs * useful))
        total_success += float(np.sum(probs))
        min_ci = min(min_ci, float(np.min(coherent_information)))
        useful_outcomes += int(np.count_nonzero(coherent_information > 0.0))

        fidelities = np.einsum("ki,kij,kj->k", np.conj(target), rho, target).real
        fidelities = np.clip(fidelities, 0.0, 1.0)
        fidelity_weight += float(np.sum(probs * fidelities))

    avg_fidelity = fidelity_weight / total_success if total_success > EPS else 0.0
    if min_ci == float("inf"):
        min_ci = 0.0
    return StrategyResult(
        rate=float(total_rate),
        success_probability=float(total_success),
        average_fidelity=float(avg_fidelity),
        min_coherent_information=float(min_ci),
        useful_outcomes=int(useful_outcomes),
    )


def local_loss_eigendecomposition(
    local_loss: np.ndarray, rank_tol: float = LOSS_RANK_TOL
) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = scipy.linalg.eigh((local_loss + local_loss.conj().T) / 2.0)
    evals = np.real(evals)
    evals[np.abs(evals) < 1.0e-14] = 0.0
    evals = np.clip(evals, 0.0, None)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if evals.size == 0 or evals[0] <= EPS:
        return evals[:0], evecs[:, :0]
    keep = evals > rank_tol * evals[0]
    return evals[keep], evecs[:, keep]


def entropy_bits_from_density_eigvalsh(rho: np.ndarray) -> float:
    evals = scipy.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    return float(entropy_bits_from_eigvals(evals[None, :])[0])


def entropy_ab_from_factorized_loss(
    weights: np.ndarray,
    norm2: float,
    loss_evals: np.ndarray,
    loss_evecs: np.ndarray,
    pair_weight_outer_sqrt: np.ndarray,
) -> float:
    rank = loss_evals.size
    if rank == 0 or norm2 <= EPS:
        return 0.0
    if rank == 1:
        return 0.0

    # Nonzero eigenvalues of D_v (L_A \otimes L_B) D_v^\dagger are equal to
    # those of the smaller environment Gram matrix below.  This is much faster
    # whenever the local environment Gram is numerically low rank, which is the
    # common case for low per-arm loss.
    left_contract = np.einsum(
        "ar,ab,as->rsb", loss_evecs.conj(), weights, loss_evecs, optimize=True
    )
    env_gram = np.einsum(
        "bt,rsb,bu->rtsu",
        loss_evecs.conj(),
        left_contract,
        loss_evecs,
        optimize=True,
    ).reshape(rank * rank, rank * rank)
    env_gram *= pair_weight_outer_sqrt
    env_gram /= norm2
    env_gram = (env_gram + env_gram.conj().T) / 2.0
    evals = scipy.linalg.eigvalsh(env_gram)
    return float(entropy_bits_from_eigvals(evals[None, :])[0])


def fidelity_from_factorized_loss(
    v: np.ndarray, target: np.ndarray, local_loss: np.ndarray, m: int, norm2: float
) -> float:
    target_weighted = (np.conj(v) * target).reshape(m, m)
    loss_applied = local_loss @ target_weighted @ local_loss.T
    fidelity = np.vdot(target_weighted, loss_applied).real / norm2
    return float(np.clip(fidelity, 0.0, 1.0))


def evaluate_strategy_factorized_loss(
    overlaps: np.ndarray,
    vec_gram: np.ndarray,
    targets: np.ndarray,
    local_loss: np.ndarray,
    m: int,
    rank_tol: float = LOSS_RANK_TOL,
) -> StrategyResult:
    dim = m * m
    scale = povm_scale(vec_gram)
    loss_evals, loss_evecs = local_loss_eigendecomposition(local_loss, rank_tol)
    pair_evals = np.multiply.outer(loss_evals, loss_evals).reshape(-1)
    pair_weight_outer_sqrt = np.sqrt(np.outer(pair_evals, pair_evals))

    total_rate = 0.0
    total_success = 0.0
    fidelity_weight = 0.0
    min_ci = float("inf")
    useful_outcomes = 0

    for v, target in zip(overlaps, targets, strict=True):
        norm2 = float(np.sum(np.abs(v) ** 2).real)
        if norm2 <= 1.0e-14:
            continue

        prob = scale * norm2 / dim
        v_matrix = v.reshape(m, m)
        weights = np.abs(v_matrix) ** 2

        rho_a = local_loss * (v_matrix @ v_matrix.conj().T)
        rho_a /= norm2
        s_a = entropy_bits_from_density_eigvalsh(rho_a)

        s_ab = entropy_ab_from_factorized_loss(
            weights,
            norm2,
            loss_evals,
            loss_evecs,
            pair_weight_outer_sqrt,
        )
        coherent_information = s_a - s_ab
        useful = max(coherent_information, 0.0)

        total_rate += prob * useful
        total_success += prob
        min_ci = min(min_ci, coherent_information)
        useful_outcomes += int(coherent_information > 0.0)
        fidelity_weight += prob * fidelity_from_factorized_loss(
            v, target, local_loss, m, norm2
        )

    avg_fidelity = fidelity_weight / total_success if total_success > EPS else 0.0
    if min_ci == float("inf"):
        min_ci = 0.0
    return StrategyResult(
        rate=float(total_rate),
        success_probability=float(total_success),
        average_fidelity=float(avg_fidelity),
        min_coherent_information=float(min_ci),
        useful_outcomes=int(useful_outcomes),
    )


def evaluate_point(
    alpha: float,
    eta: float,
    m: int,
    family: str,
    rank_tol: float = LOSS_RANK_TOL,
) -> dict[str, StrategyResult]:
    coeffs, _labels, targets = sparse_measurement_coefficients(m, family)
    gram = coherent_pair_gram(alpha, eta, m)
    local_loss = local_loss_coherence_matrix(alpha, eta, m)
    std_overlaps, std_gram = standard_overlaps_after_vacuum_subtraction(
        coeffs, gram, alpha, eta
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


def _evaluate_alpha_task(
    args: tuple[float, str, int, float, float]
) -> tuple[float, dict[str, StrategyResult]]:
    alpha, family, m, eta, rank_tol = args
    key = round(float(alpha), 12)
    return key, evaluate_point(float(alpha), eta, m, family, rank_tol=rank_tol)


def optimize_alpha(
    family: str,
    m: int,
    loss_db: float,
    alpha_min: float,
    alpha_max: float,
    coarse_points: int,
    refine_points: int,
    executor: Executor | None = None,
    rank_tol: float = LOSS_RANK_TOL,
) -> dict[str, SweepBest]:
    eta = 10.0 ** (-loss_db / 10.0)
    coarse_alphas = np.linspace(alpha_min, alpha_max, coarse_points)
    cache: dict[float, dict[str, StrategyResult]] = {}

    def evaluate_alphas(alphas: Iterable[float]) -> None:
        missing = []
        for alpha in alphas:
            key = round(float(alpha), 12)
            if key not in cache:
                missing.append(float(alpha))
        if not missing:
            return
        if executor is None or len(missing) == 1:
            for alpha in missing:
                key, result = _evaluate_alpha_task((alpha, family, m, eta, rank_tol))
                cache[key] = result
        else:
            tasks = [(alpha, family, m, eta, rank_tol) for alpha in missing]
            for key, result in executor.map(_evaluate_alpha_task, tasks):
                cache[key] = result

    def eval_alpha(alpha: float) -> dict[str, StrategyResult]:
        key = round(float(alpha), 12)
        if key not in cache:
            evaluate_alphas([alpha])
        return cache[key]

    evaluate_alphas(coarse_alphas)

    best: dict[str, tuple[float, StrategyResult]] = {}
    for strategy in ["Standard", "YKL"]:
        pairs = [
            (alpha, eval_alpha(float(alpha))[strategy]) for alpha in coarse_alphas
        ]
        best[strategy] = max(pairs, key=lambda item: item[1].rate)

    step = float(coarse_alphas[1] - coarse_alphas[0]) if coarse_points > 1 else 0.1
    for strategy, (alpha0, _result0) in list(best.items()):
        lo = max(alpha_min, alpha0 - 2.0 * step)
        hi = min(alpha_max, alpha0 + 2.0 * step)
        if math.isclose(alpha0, alpha_min):
            hi = min(alpha_max, alpha0 + 4.0 * step)
        if math.isclose(alpha0, alpha_max):
            lo = max(alpha_min, alpha0 - 4.0 * step)
        refine_alphas = np.linspace(lo, hi, refine_points)
        evaluate_alphas(refine_alphas)
        pairs = [
            (alpha, eval_alpha(float(alpha))[strategy]) for alpha in refine_alphas
        ]
        best[strategy] = max(
            [best[strategy], *pairs], key=lambda item: item[1].rate
        )

    return {
        strategy: SweepBest(
            family=family,
            m=m,
            loss_db=loss_db,
            eta=eta,
            strategy=strategy,
            best_alpha=float(alpha),
            result=result,
        )
        for strategy, (alpha, result) in best.items()
    }


def run_sweep(
    family: str,
    m_values: Iterable[int],
    losses_db: Iterable[float],
    alpha_min: float,
    alpha_max: float,
    coarse_points: int,
    refine_points: int,
    workers: int = 1,
    rank_tol: float = LOSS_RANK_TOL,
) -> list[SweepBest]:
    rows: list[SweepBest] = []
    executor_cm = ThreadPoolExecutor(max_workers=workers) if workers and workers > 1 else None
    try:
        if executor_cm is not None:
            executor_cm.__enter__()
        for m in m_values:
            for loss_db in losses_db:
                print(f"Running family={family}, M={m}, loss={loss_db:.2f} dB")
                best = optimize_alpha(
                    family=family,
                    m=m,
                    loss_db=loss_db,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
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
                        f"alpha={item.best_alpha:.3f}, "
                        f"Psucc={item.result.success_probability:.4f}, "
                        f"Favg={item.result.average_fidelity:.4f}"
                    )
    finally:
        if executor_cm is not None:
            executor_cm.__exit__(None, None, None)
    return rows


def write_csv(rows: list[SweepBest], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "M",
                "family",
                "qubits_per_side",
                "loss_db",
                "eta",
                "strategy",
                "best_alpha",
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
                    row.family,
                    int(round(math.log2(row.m))),
                    f"{row.loss_db:.6g}",
                    f"{row.eta:.12g}",
                    row.strategy,
                    f"{row.best_alpha:.12g}",
                    f"{row.result.rate:.12g}",
                    f"{row.result.success_probability:.12g}",
                    f"{row.result.average_fidelity:.12g}",
                    f"{row.result.min_coherent_information:.12g}",
                    row.result.useful_outcomes,
                ]
            )


def write_json(rows: list[SweepBest], path: Path) -> None:
    data = []
    for row in rows:
        data.append(
            {
                "M": row.m,
                "family": row.family,
                "qubits_per_side": int(round(math.log2(row.m))),
                "loss_db": row.loss_db,
                "eta": row.eta,
                "strategy": row.strategy,
                "best_alpha": row.best_alpha,
                "hashing_bound_bits_per_attempt": row.result.rate,
                "success_probability": row.result.success_probability,
                "average_target_fidelity": row.result.average_fidelity,
                "min_coherent_information": row.result.min_coherent_information,
                "useful_outcomes": row.result.useful_outcomes,
            }
        )
    path.write_text(json.dumps(data, indent=2))


def plot_rates(rows: list[SweepBest], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colors = {2: "#2364AA", 4: "#3DA35D", 8: "#D65A31", 16: "#7B2CBF"}
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
                [row.result.rate for row in subset],
                marker=markers[strategy],
                linestyle=linestyles[strategy],
                color=colors[m],
                label=f"{m}-PSK {strategy}",
                linewidth=1.8,
                markersize=4.5,
            )

    ax.set_xlabel("Per-arm channel loss to Charlie (dB)")
    ax.set_ylabel("Optimized weighted hashing bound (bits/attempt)")
    positive_rates = [row.result.rate for row in rows if row.result.rate > 0.0]
    if positive_rates:
        ax.set_yscale("log")
        ax.set_ylim(min(positive_rates) * 0.75, max(positive_rates) * 1.25)
    ax.set_xlim(-0.05, 3.05)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_strategy_rates(rows: list[SweepBest], strategy: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colors = {2: "#2364AA", 4: "#3DA35D", 8: "#D65A31", 16: "#7B2CBF"}
    markers = {2: "o", 4: "s", 8: "^", 16: "D"}
    subset_all = [row for row in rows if row.strategy == strategy]

    for m in sorted({row.m for row in subset_all}):
        subset = sorted([row for row in subset_all if row.m == m], key=lambda r: r.loss_db)
        ax.plot(
            [row.loss_db for row in subset],
            [row.result.rate for row in subset],
            marker=markers.get(m, "o"),
            linestyle="-",
            color=colors.get(m, "#333333"),
            label=f"{m}-PSK",
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
    ax.set_title(f"{strategy} POVM")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_best_alpha(rows: list[SweepBest], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    colors = {2: "#2364AA", 4: "#3DA35D", 8: "#D65A31", 16: "#7B2CBF"}
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
                [row.best_alpha for row in subset],
                marker=markers[strategy],
                linestyle=linestyles[strategy],
                color=colors[m],
                label=f"{m}-PSK {strategy}",
                linewidth=1.8,
                markersize=4.5,
            )
    ax.set_xlabel("Per-arm channel loss to Charlie (dB)")
    ax.set_ylabel("Optimized launch amplitude alpha")
    ax.set_xlim(-0.05, 3.05)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def latex_float(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def write_report_tables(rows: list[SweepBest], path: Path) -> None:
    by_key = {(row.m, row.loss_db, row.strategy): row for row in rows}
    losses = sorted({row.loss_db for row in rows})
    m_values = sorted({row.m for row in rows})

    lines: list[str] = []
    lines.append("% Auto-generated by mpsk_ghz_hashing.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Optimized weighted hashing bound, in bits per attempt.}")
    lines.append("\\label{tab:hashing-results}")
    lines.append("\\scriptsize")
    colspec = "cc" + "rr" * len(m_values)
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\hline")
    header = ["Loss (dB)", "$\\eta$"]
    for m in m_values:
        header.extend([f"{m}-PSK Std.", f"{m}-PSK YKL"])
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")
    for loss in losses:
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
    lines.append("\\caption{Optimized launch amplitudes $\\alpha$ for the same sweep.}")
    lines.append("\\label{tab:alpha-results}")
    lines.append("\\scriptsize")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")
    for loss in losses:
        eta = by_key[(m_values[0], loss, "Standard")].eta
        row = [latex_float(loss, 1), latex_float(eta, 4)]
        for m in m_values:
            row.append(latex_float(by_key[(m, loss, "Standard")].best_alpha, 3))
            row.append(latex_float(by_key[(m, loss, "YKL")].best_alpha, 3))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="mpsk_hashing_results")
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=10.0)
    parser.add_argument("--coarse-points", type=int, default=100)
    parser.add_argument("--refine-points", type=int, default=51)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker threads for alpha evaluations. Use 4-8 for 16-PSK.",
    )
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
        default="ghz",
        help="Measurement target family: rank-2 GHZ pairs or full M-branch Bell states.",
    )
    parser.add_argument(
        "--losses-db",
        default="0,0.5,1,1.5,2,2.5,3",
        help="Comma-separated per-arm losses in dB.",
    )
    parser.add_argument("--loss-min", type=float, default=None)
    parser.add_argument("--loss-max", type=float, default=None)
    parser.add_argument("--loss-step", type=float, default=None)
    parser.add_argument(
        "--m-values",
        default="2,4,8",
        help="Comma-separated PSK orders. Values should be powers of two, e.g. 2,4,8,16.",
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
        if m <= 1 or m & (m - 1):
            raise ValueError(f"M={m} is not a power of two greater than one.")

    try:
        rows = run_sweep(
            family=args.family,
            m_values=m_values,
            losses_db=losses_db,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            coarse_points=args.coarse_points,
            refine_points=args.refine_points,
            workers=args.workers,
            rank_tol=args.loss_rank_tol,
        )

        stem = f"mpsk_{args.family}_hashing_summary"
        write_csv(rows, outdir / f"{stem}.csv")
        write_json(rows, outdir / f"{stem}.json")
        write_report_tables(rows, outdir / "report_tables.tex")
        plot_rates(rows, outdir / "hash_bound_vs_loss.png")
        plot_strategy_rates(rows, "Standard", outdir / "standard_hash_bound_vs_loss.png")
        plot_strategy_rates(rows, "YKL", outdir / "ykl_hash_bound_vs_loss.png")
        plot_best_alpha(rows, outdir / "best_alpha_vs_loss.png")
        print(f"Wrote results to {outdir}")
    finally:
        if thread_limiter is not None:
            thread_limiter.__exit__(None, None, None)


if __name__ == "__main__":
    main()
