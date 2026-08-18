#!/usr/bin/env python3
"""Check raw-label SRM vs vacuum-omit SRM for QAM across loss."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
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
import scipy.optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare_schmidt_bell_povm_qam import raw_bell_codeword_overlaps
from mpsk_ghz_hashing import (
    LOSS_RANK_TOL,
    evaluate_strategy_factorized_loss,
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
    "loss_db",
    "receiver",
    "best_d",
    "hashing_bound",
    "success_probability",
    "average_target_fidelity",
    "useful_outcomes",
]


@dataclass(frozen=True)
class ReceiverOptimum:
    m: int
    loss_db: float
    receiver: str
    best_d: float
    rate: float
    success_probability: float
    average_fidelity: float
    useful_outcomes: int


def normalized_rows(rows: np.ndarray, eps: float = 1.0e-14) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1)
    out = np.zeros_like(rows)
    keep = norms > eps
    out[keep] = rows[keep] / norms[keep, None]
    return out


def evaluate_receiver(m: int, loss_db: float, spacing: float, receiver: str) -> tuple[float, float, float, int]:
    eta = 10.0 ** (-loss_db / 10.0)
    local_amps = qam_constellation(m, spacing)
    gram, vac_overlaps = coherent_pair_gram_from_amplitudes(local_amps, eta)
    local_loss = local_loss_coherence_from_amplitudes(local_amps, eta)
    coeffs, _labels, _bell_targets = sparse_measurement_coefficients(m, "bell")

    if receiver == "vacuum_omit_srm":
        overlaps, vec_gram = standard_overlaps_after_vacuum_subtraction(
            coeffs, gram, vac_overlaps
        )
    elif receiver == "raw_label_srm":
        overlaps, vec_gram = raw_bell_codeword_overlaps(coeffs, gram)
    else:
        raise ValueError(f"Unknown receiver {receiver!r}")

    srm_overlaps, srm_gram = ykl_square_root_measurement(overlaps, vec_gram)
    targets = normalized_rows(srm_overlaps)
    result = evaluate_strategy_factorized_loss(
        srm_overlaps,
        srm_gram,
        targets,
        local_loss,
        m,
        rank_tol=LOSS_RANK_TOL,
    )
    return (
        float(result.rate),
        float(result.success_probability),
        float(result.average_fidelity),
        int(result.useful_outcomes),
    )


def optimize_receiver(
    m: int,
    loss_db: float,
    receiver: str,
    spacing_min: float,
    spacing_max: float,
    coarse_points: int,
) -> ReceiverOptimum:
    cache: dict[float, tuple[float, float, float, int]] = {}

    def evaluate(spacing: float) -> tuple[float, float, float, int]:
        key = round(float(spacing), 12)
        if key not in cache:
            cache[key] = evaluate_receiver(m, loss_db, float(spacing), receiver)
        return cache[key]

    coarse = np.linspace(spacing_min, spacing_max, coarse_points)
    coarse_rates = np.array([evaluate(float(d))[0] for d in coarse])
    best_idx = int(np.argmax(coarse_rates))
    best_d = float(coarse[best_idx])

    if best_idx == 0:
        bracket = (float(coarse[0]), float(coarse[min(4, len(coarse) - 1)]))
    elif best_idx == len(coarse) - 1:
        bracket = (float(coarse[max(0, len(coarse) - 5)]), float(coarse[-1]))
    else:
        bracket = (
            float(coarse[max(0, best_idx - 2)]),
            float(coarse[min(len(coarse) - 1, best_idx + 2)]),
        )

    def objective(spacing: float) -> float:
        rate = evaluate(float(spacing))[0]
        if math.isnan(rate):
            return 1.0e9
        return -rate

    opt = scipy.optimize.minimize_scalar(
        objective,
        bounds=bracket,
        method="bounded",
        options={"xatol": 1.0e-5, "maxiter": 80},
    )
    if opt.success:
        candidate_d = float(opt.x)
        if evaluate(candidate_d)[0] > evaluate(best_d)[0]:
            best_d = candidate_d

    rate, success, fidelity, useful = evaluate(best_d)
    return ReceiverOptimum(
        m=m,
        loss_db=loss_db,
        receiver=receiver,
        best_d=best_d,
        rate=rate,
        success_probability=success,
        average_fidelity=fidelity,
        useful_outcomes=useful,
    )


def run_point(args: tuple[int, float, float, float, int]) -> list[ReceiverOptimum]:
    m, loss_db, spacing_min, spacing_max, coarse_points = args
    return [
        optimize_receiver(m, loss_db, receiver, spacing_min, spacing_max, coarse_points)
        for receiver in ("vacuum_omit_srm", "raw_label_srm")
    ]


def write_csv(rows: list[ReceiverOptimum], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.m, item.loss_db, item.receiver)):
            writer.writerow(
                {
                    "M": row.m,
                    "loss_db": f"{row.loss_db:.12g}",
                    "receiver": row.receiver,
                    "best_d": f"{row.best_d:.12g}",
                    "hashing_bound": f"{row.rate:.12g}",
                    "success_probability": f"{row.success_probability:.12g}",
                    "average_target_fidelity": f"{row.average_fidelity:.12g}",
                    "useful_outcomes": row.useful_outcomes,
                }
            )


def write_delta_csv(rows: list[ReceiverOptimum], path: Path) -> None:
    by_key = {(row.m, row.loss_db, row.receiver): row for row in rows}
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "M",
                "loss_db",
                "vacuum_rate",
                "raw_rate",
                "raw_minus_vacuum",
                "vacuum_d",
                "raw_d",
            ],
        )
        writer.writeheader()
        for m, loss_db in sorted({(row.m, row.loss_db) for row in rows}):
            vac = by_key[(m, loss_db, "vacuum_omit_srm")]
            raw = by_key[(m, loss_db, "raw_label_srm")]
            writer.writerow(
                {
                    "M": m,
                    "loss_db": f"{loss_db:.12g}",
                    "vacuum_rate": f"{vac.rate:.12g}",
                    "raw_rate": f"{raw.rate:.12g}",
                    "raw_minus_vacuum": f"{raw.rate - vac.rate:.12g}",
                    "vacuum_d": f"{vac.best_d:.12g}",
                    "raw_d": f"{raw.best_d:.12g}",
                }
            )


def plot_delta(delta_csv: Path, path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(delta_csv)
    colors = {2: "#4C78A8", 4: "#F58518", 8: "#54A24B", 16: "#B279A2"}
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for m in sorted(df["M"].unique()):
        sub = df[df["M"] == m].sort_values("loss_db")
        ax.plot(
            sub["loss_db"],
            sub["raw_minus_vacuum"],
            marker="o",
            markersize=3.0,
            linewidth=1.8,
            color=colors.get(int(m)),
            label=f"{int(m)}-QAM",
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("channel loss (dB)")
    ax.set_ylabel("raw-label SRM minus vacuum-omit SRM")
    ax.set_title("Optimized hashing-bound difference")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_losses(text: str) -> list[float]:
    if ":" in text:
        start, stop, step = [float(part) for part in text.split(":")]
        count = int(round((stop - start) / step))
        return [round(start + idx * step, 12) for idx in range(count + 1)]
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results" / "raw_label_vs_vacuum_omit_srm_qam",
    )
    parser.add_argument("--ms", default="2,4,8,16")
    parser.add_argument("--losses-db", default="0:10:0.5")
    parser.add_argument("--spacing-min", type=float, default=0.02)
    parser.add_argument("--spacing-max", type=float, default=6.0)
    parser.add_argument("--coarse-points", type=int, default=41)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ms = [int(part.strip()) for part in args.ms.split(",") if part.strip()]
    losses = parse_losses(args.losses_db)
    jobs = [
        (m, loss_db, args.spacing_min, args.spacing_max, args.coarse_points)
        for m in ms
        for loss_db in losses
    ]
    rows: list[ReceiverOptimum] = []
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.outdir / "raw_label_vs_vacuum_omit_srm_optima.csv"
    delta_csv = args.outdir / "raw_label_minus_vacuum_omit_srm.csv"

    if args.workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_point, job): job for job in jobs}
            for future in as_completed(futures):
                batch = future.result()
                rows.extend(batch)
                write_csv(rows, raw_csv)
                write_delta_csv(rows, delta_csv)
                by_receiver = {row.receiver: row for row in batch}
                vac = by_receiver["vacuum_omit_srm"]
                raw = by_receiver["raw_label_srm"]
                print(
                    f"M={vac.m} loss={vac.loss_db:g}: "
                    f"vac={vac.rate:.8g}, raw={raw.rate:.8g}, "
                    f"diff={raw.rate - vac.rate:+.3g}"
                )
    else:
        for job in jobs:
            batch = run_point(job)
            rows.extend(batch)
            write_csv(rows, raw_csv)
            write_delta_csv(rows, delta_csv)

    write_csv(rows, raw_csv)
    write_delta_csv(rows, delta_csv)
    plot_delta(delta_csv, args.outdir / "raw_label_minus_vacuum_omit_srm.png")
    print(raw_csv)
    print(delta_csv)
    print(args.outdir / "raw_label_minus_vacuum_omit_srm.png")


if __name__ == "__main__":
    main()
