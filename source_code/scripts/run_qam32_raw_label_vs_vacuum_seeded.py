#!/usr/bin/env python3
"""Seeded 32-QAM raw-label SRM vs vacuum-omit SRM comparison.

32-QAM uses 1024 optical labels, so each SRM evaluation is expensive.  This
script reuses the previously refined vacuum-omit 32-QAM branch optima and runs
a local candidate scan for raw-label SRM around the same branch seeds.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
from check_raw_label_vs_vacuum_omit_srm_qam import evaluate_receiver


VACUUM_SOURCE = ROOT / "results" / "refined_qam_branches" / "32qam_seeded_refined_global_optima.csv"
BRANCH_SOURCE = ROOT / "results" / "refined_qam_branches" / "qam_refined_global_rate_vs_loss_combined_dense_transitions_with_32qam.csv"
OUTDIR = ROOT / "results" / "raw_label_vs_vacuum_omit_srm_qam32_seeded"
OUTCSV = OUTDIR / "qam32_raw_label_vs_vacuum_seeded.csv"


@dataclass(frozen=True)
class Candidate:
    loss_db: float
    d: float
    seed_source: str


@dataclass(frozen=True)
class EvalResult:
    loss_db: float
    d: float
    rate: float
    success_probability: float
    average_target_fidelity: float
    useful_outcomes: int
    seed_source: str
    seconds: float


def target_losses(vacuum: pd.DataFrame) -> list[float]:
    losses = sorted(float(x) for x in vacuum["loss_db"].unique())
    # Keep the same 0--1.5 dB support as the refined 32-QAM vacuum data.
    return [x for x in losses if 0.0 <= x <= 1.5]


def load_vacuum() -> pd.DataFrame:
    vacuum = pd.read_csv(VACUUM_SOURCE)
    vacuum = vacuum[vacuum["is_global_maximum"].astype(int) == 1].copy()
    vacuum["loss_db"] = vacuum["loss_db"].astype(float).round(6)
    vacuum = vacuum.sort_values("loss_db").drop_duplicates("loss_db", keep="last")
    return vacuum


def branch_seed_map(losses: list[float]) -> dict[float, list[float]]:
    seeds: dict[float, list[float]] = {round(loss, 6): [] for loss in losses}
    if not BRANCH_SOURCE.exists():
        return seeds
    branches = pd.read_csv(BRANCH_SOURCE)
    branches = branches[branches["M"].astype(int) == 32].copy()
    branches["loss_db"] = branches["loss_db"].astype(float).round(6)
    for loss, group in branches.groupby("loss_db"):
        key = round(float(loss), 6)
        if key not in seeds:
            continue
        for d in group["spacing_d"].astype(float):
            if math.isfinite(d) and d > 0:
                seeds[key].append(float(d))
    return seeds


def make_candidates(vacuum: pd.DataFrame) -> dict[float, list[Candidate]]:
    losses = target_losses(vacuum)
    branch_seeds = branch_seed_map(losses)
    out: dict[float, list[Candidate]] = {}
    for loss in losses:
        key = round(loss, 6)
        row = vacuum[vacuum["loss_db"].round(6) == key].iloc[0]
        seeds = list(branch_seeds.get(key, []))
        d_vac = float(row["spacing_d"])
        if math.isfinite(d_vac) and d_vac > 0:
            seeds.append(d_vac)

        if key == 0.0:
            seeds = [4.0]

        candidates: dict[float, str] = {}
        for seed in seeds:
            if not math.isfinite(seed) or seed <= 0:
                continue
            multipliers = [1.0]
            if seed > 1.0:
                multipliers = [0.97, 1.0, 1.03]
            else:
                multipliers = [0.94, 1.0, 1.06]
            for mult in multipliers:
                d = round(float(seed * mult), 10)
                if d > 0:
                    candidates.setdefault(d, "branch_seed")

        out[key] = [Candidate(key, d, source) for d, source in sorted(candidates.items())]
    return out


def evaluate_candidate(candidate: Candidate) -> EvalResult:
    start = time.time()
    if candidate.loss_db == 0.0:
        # At zero loss, both measurements can reach log2(32)=5 in the
        # large-spacing limit; keep the saved high-d representative.
        return EvalResult(candidate.loss_db, candidate.d, 5.0, 1.0, 1.0, 1024, candidate.seed_source, 0.0)
    rate, success, fidelity, useful = evaluate_receiver(
        32, candidate.loss_db, candidate.d, "raw_label_srm"
    )
    return EvalResult(
        candidate.loss_db,
        candidate.d,
        rate,
        success,
        fidelity,
        useful,
        candidate.seed_source,
        time.time() - start,
    )


def write_rows(vacuum: pd.DataFrame, best: dict[float, EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "M",
        "loss_db",
        "vacuum_rate",
        "raw_rate",
        "raw_minus_vacuum",
        "vacuum_d",
        "raw_d",
        "raw_success_probability",
        "raw_average_target_fidelity",
        "raw_useful_outcomes",
        "raw_candidate_seconds",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for _, row in vacuum.sort_values("loss_db").iterrows():
            loss = round(float(row["loss_db"]), 6)
            if loss not in best:
                continue
            raw = best[loss]
            vac_rate = float(row["hashing_bound_bits_per_attempt"])
            vac_d = float(row["spacing_d"]) if math.isfinite(float(row["spacing_d"])) else 4.0
            if loss == 0.0:
                vac_rate = 5.0
                vac_d = 4.0
            writer.writerow(
                {
                    "M": 32,
                    "loss_db": f"{loss:.12g}",
                    "vacuum_rate": f"{vac_rate:.12g}",
                    "raw_rate": f"{raw.rate:.12g}",
                    "raw_minus_vacuum": f"{raw.rate - vac_rate:.12g}",
                    "vacuum_d": f"{vac_d:.12g}",
                    "raw_d": f"{raw.d:.12g}",
                    "raw_success_probability": f"{raw.success_probability:.12g}",
                    "raw_average_target_fidelity": f"{raw.average_target_fidelity:.12g}",
                    "raw_useful_outcomes": raw.useful_outcomes,
                    "raw_candidate_seconds": f"{raw.seconds:.3f}",
                    "note": "raw d selected from seeded local branch candidates",
                }
            )


def main() -> None:
    vacuum = load_vacuum()
    candidates_by_loss = make_candidates(vacuum)
    best: dict[float, EvalResult] = {}
    pending = [candidate for group in candidates_by_loss.values() for candidate in group]

    # Keep memory pressure under control for 1024-dimensional SRMs.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(evaluate_candidate, candidate): candidate for candidate in pending}
        completed_by_loss: dict[float, list[EvalResult]] = {loss: [] for loss in candidates_by_loss}
        for future in as_completed(futures):
            result = future.result()
            completed_by_loss[result.loss_db].append(result)
            group = completed_by_loss[result.loss_db]
            needed = len(candidates_by_loss[result.loss_db])
            if len(group) == needed:
                selected = max(group, key=lambda item: item.rate)
                best[result.loss_db] = selected
                write_rows(vacuum, best, OUTCSV)
                print(
                    f"loss={result.loss_db:g}: raw={selected.rate:.8g}, "
                    f"d={selected.d:.6g}, candidates={needed}"
                )

    write_rows(vacuum, best, OUTCSV)
    print(OUTCSV)


if __name__ == "__main__":
    main()
