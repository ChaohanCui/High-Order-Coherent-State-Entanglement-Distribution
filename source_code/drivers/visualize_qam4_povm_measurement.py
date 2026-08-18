#!/usr/bin/env python3
"""Visualize an optimized 4-QAM POVM and its lack of simple symmetry."""

from __future__ import annotations

import argparse
import os
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
import scipy.optimize

import optimize_qam4_general_povm as opt
from mpsk_ghz_hashing import entropy_bits, sparse_measurement_coefficients


def set_m(m: int) -> None:
    opt.M = m
    opt.DIM = m * m


def bell_probabilities(rho_ab: np.ndarray) -> np.ndarray:
    _coeffs, _labels, targets = sparse_measurement_coefficients(opt.M, "bell")
    return np.einsum("ki,ij,kj->k", targets.conj(), rho_ab, targets).real


def outcome_table(a: np.ndarray, problem: opt.Problem) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    bell_matrix = []
    for idx, row in enumerate(a):
        kernel, _overlaps = opt.row_kernel(row, problem.r)
        prob = float(np.real(np.trace(kernel)) / opt.DIM)
        if prob <= 1.0e-14:
            continue

        tau_ab = (kernel * problem.pair_loss) / opt.DIM
        tau_ab = (tau_ab + tau_ab.conj().T) / 2.0
        rho_ab = tau_ab / prob

        tau_a = np.zeros((opt.M, opt.M), dtype=complex)
        for x in range(opt.M):
            for xp in range(opt.M):
                tau_a[x, xp] = sum(
                    tau_ab[x * opt.M + y, xp * opt.M + y] for y in range(opt.M)
                )
        tau_a = (tau_a + tau_a.conj().T) / 2.0
        rho_a = tau_a / prob

        coherent_information = entropy_bits(rho_a) - entropy_bits(rho_ab)
        bells = np.clip(bell_probabilities(rho_ab), 0.0, 1.0)
        top = np.sort(bells)[::-1]
        rows.append(
            {
                "outcome": idx,
                "probability": prob,
                "coherent_information": coherent_information,
                "rate_contribution": prob * max(coherent_information, 0.0),
                "nearest_bell_label": int(np.argmax(bells)),
                "nearest_bell_fidelity": float(top[0]),
                "second_bell_fidelity": float(top[1]),
                "bell_participation_ratio": float(1.0 / np.sum(bells**2)),
                "row_norm2": float(np.vdot(row, row).real),
            }
        )
        bell_matrix.append(bells)
    return pd.DataFrame(rows), np.asarray(bell_matrix)


def branch_permutation_matrix(mapping: list[int]) -> np.ndarray:
    dim = opt.M * opt.M
    p = np.zeros((dim, dim), dtype=complex)
    for old, new in enumerate(mapping):
        p[new, old] = 1.0
    return p


def local_xor_mapping(xa: int, xb: int) -> list[int]:
    return [(a ^ xa) * opt.M + (b ^ xb) for a in range(opt.M) for b in range(opt.M)]


def swap_mapping() -> list[int]:
    return [b * opt.M + a for a in range(opt.M) for b in range(opt.M)]


def support_unitary(problem: opt.Problem, mapping: list[int]) -> np.ndarray:
    p = branch_permutation_matrix(mapping)
    return problem.r @ p @ np.linalg.inv(problem.r)


def symmetry_scores(a: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed = a @ u.conj().T
    norms = np.sum(np.abs(a) ** 2, axis=1)
    transformed_norms = np.sum(np.abs(transformed) ** 2, axis=1)
    overlaps = np.abs(transformed @ a.conj().T) ** 2
    denom = transformed_norms[:, None] * norms[None, :]
    scores = overlaps / np.maximum(denom, 1.0e-30)
    weight_mismatch = np.abs(transformed_norms[:, None] - norms[None, :]) / np.maximum(
        transformed_norms[:, None], 1.0e-30
    )
    return scores, weight_mismatch


def symmetry_diagnostics(a: np.ndarray, problem: opt.Problem) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    labels: list[str] = []
    best_overlap_rows = []
    summary_rows = []

    symmetries: list[tuple[str, list[int]]] = []
    for xa in range(opt.M):
        for xb in range(opt.M):
            if xa == 0 and xb == 0:
                continue
            symmetries.append((f"xor {xa},{xb}", local_xor_mapping(xa, xb)))
    symmetries.append(("swap A,B", swap_mapping()))

    for label, mapping in symmetries:
        u = support_unitary(problem, mapping)
        scores, weight_mismatch = symmetry_scores(a, u)
        best_idx = np.argmax(scores, axis=1)
        best = scores[np.arange(scores.shape[0]), best_idx]
        best_weight_mismatch = weight_mismatch[np.arange(scores.shape[0]), best_idx]

        assign_i, assign_j = scipy.optimize.linear_sum_assignment(-scores)
        assigned_scores = scores[assign_i, assign_j]
        assigned_weight_mismatch = weight_mismatch[assign_i, assign_j]

        labels.append(label)
        best_overlap_rows.append(best)
        summary_rows.append(
            {
                "symmetry": label,
                "mean_rowwise_best_overlap": float(np.mean(best)),
                "min_rowwise_best_overlap": float(np.min(best)),
                "median_rowwise_best_overlap": float(np.median(best)),
                "mean_assigned_overlap": float(np.mean(assigned_scores)),
                "min_assigned_overlap": float(np.min(assigned_scores)),
                "median_assigned_overlap": float(np.median(assigned_scores)),
                "mean_rowwise_weight_mismatch": float(np.mean(best_weight_mismatch)),
                "max_rowwise_weight_mismatch": float(np.max(best_weight_mismatch)),
                "mean_assigned_weight_mismatch": float(np.mean(assigned_weight_mismatch)),
                "max_assigned_weight_mismatch": float(np.max(assigned_weight_mismatch)),
            }
        )

    return pd.DataFrame(summary_rows), np.asarray(best_overlap_rows), labels


def make_overview_plot(
    table: pd.DataFrame,
    bell_matrix: np.ndarray,
    symmetry_heatmap: np.ndarray,
    symmetry_labels: list[str],
    out_path: Path,
    figure_title: str | None = None,
) -> None:
    order = np.lexsort(
        (
            -table["rate_contribution"].to_numpy(),
            table["nearest_bell_label"].to_numpy(),
        )
    )
    sorted_table = table.iloc[order].reset_index(drop=True)
    sorted_bells = bell_matrix[order]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    if figure_title:
        fig.suptitle(figure_title)

    im0 = axes[0, 0].imshow(sorted_bells, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Bell/coset fingerprint per outcome")
    axes[0, 0].set_xlabel("Bell/coset label")
    axes[0, 0].set_ylabel("POVM outcome, sorted")
    axes[0, 0].set_xticks(range(opt.M * opt.M))
    axes[0, 0].tick_params(axis="x", labelsize=7)
    fig.colorbar(im0, ax=axes[0, 0], label="conditional Bell fidelity")

    useful = table[table["coherent_information"] > 0.0]
    counts = useful["nearest_bell_label"].value_counts().reindex(range(opt.M * opt.M), fill_value=0)
    axes[0, 1].bar(counts.index, counts.values, color="#4C78A8")
    axes[0, 1].axhline(len(table) / (opt.M * opt.M), color="black", linestyle=":", linewidth=1.2)
    axes[0, 1].set_title("Nearest Bell-label counts")
    axes[0, 1].set_xlabel("Bell/coset label")
    axes[0, 1].set_ylabel("number of useful outcomes")
    axes[0, 1].set_xticks(range(opt.M * opt.M))
    axes[0, 1].grid(True, axis="y", alpha=0.25)

    scatter = axes[1, 0].scatter(
        table["probability"],
        table["coherent_information"],
        c=table["nearest_bell_fidelity"],
        s=900.0 * np.maximum(table["rate_contribution"], 0.0006),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1, 0].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1, 0].set_title("Outcome probability vs coherent information")
    axes[1, 0].set_xlabel("outcome probability")
    axes[1, 0].set_ylabel("coherent information")
    axes[1, 0].grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=axes[1, 0], label="nearest Bell fidelity")

    im3 = axes[1, 1].imshow(symmetry_heatmap, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Best self-match after symmetry action")
    axes[1, 1].set_xlabel("POVM outcome")
    axes[1, 1].set_ylabel("symmetry")
    axes[1, 1].set_yticks(range(len(symmetry_labels)))
    axes[1, 1].set_yticklabels(symmetry_labels, fontsize=7)
    fig.colorbar(im3, ax=axes[1, 1], label="best normalized effect overlap")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "results/global_povm_search_qam4_0p25_d_scale_narrow_0p91_0p94/"
            "best_matrices/best_M4_loss_0.25_scale_0p93_outcomes_32.npz"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/qam4_measurement_visualization"),
    )
    args = parser.parse_args()

    data = np.load(args.matrix, allow_pickle=True)
    a = data["A"]
    m = int(data["M"])
    loss_db = float(data["loss_db"])
    spacing = float(data["spacing_d"])
    previous_rate = float(data["previous_ykl_rate"])
    set_m(m)

    args.outdir.mkdir(parents=True, exist_ok=True)
    problem = opt.build_problem(loss_db, spacing, previous_rate)
    table, bell_matrix = outcome_table(a, problem)
    symmetry_summary, symmetry_heatmap, symmetry_labels = symmetry_diagnostics(a, problem)

    table.to_csv(args.outdir / "optimized_povm_outcome_fingerprints.csv", index=False)
    pd.DataFrame(
        bell_matrix,
        columns=[f"bell_{idx}" for idx in range(m * m)],
    ).to_csv(args.outdir / "optimized_povm_bell_fidelity_matrix.csv", index=False)
    symmetry_summary.to_csv(args.outdir / "optimized_povm_symmetry_summary.csv", index=False)
    pd.DataFrame(symmetry_heatmap, index=symmetry_labels).to_csv(
        args.outdir / "optimized_povm_symmetry_best_overlap_heatmap.csv"
    )
    make_overview_plot(
        table,
        bell_matrix,
        symmetry_heatmap,
        symmetry_labels,
        args.outdir / "optimized_povm_measurement_overview.png",
    )

    print(f"Loaded {args.matrix}")
    print(f"M={m}, loss_db={loss_db:g}, spacing_d={spacing:.12g}")
    print(f"rate from table={table['rate_contribution'].sum():.12g}")
    print("\nOutcome summary:")
    print(
        table[
            [
                "probability",
                "coherent_information",
                "rate_contribution",
                "nearest_bell_fidelity",
                "bell_participation_ratio",
            ]
        ]
        .describe()
        .to_string()
    )
    print("\nSymmetry summary:")
    print(
        symmetry_summary[
            [
                "symmetry",
                "mean_rowwise_best_overlap",
                "min_rowwise_best_overlap",
                "mean_assigned_overlap",
                "min_assigned_overlap",
                "mean_assigned_weight_mismatch",
            ]
        ]
        .to_string(index=False)
    )
    print(args.outdir)


if __name__ == "__main__":
    main()
