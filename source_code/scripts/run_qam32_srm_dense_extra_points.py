#!/usr/bin/env python3
"""Focused extra 32-QAM SRM points for low loss and transition regions."""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_raw_label_vs_vacuum_omit_srm_qam import normalized_rows
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


COARSE_SOURCE = (
    ROOT
    / "results"
    / "raw_label_vs_vacuum_omit_srm_qam32_seeded"
    / "qam32_raw_label_vs_vacuum_seeded.csv"
)
BRANCH_SOURCE = (
    ROOT
    / "results"
    / "refined_qam_branches"
    / "qam_refined_global_rate_vs_loss_combined_dense_transitions_with_32qam.csv"
)
OUTDIR = ROOT / "results" / "raw_label_vs_vacuum_omit_srm_qam32_dense_extra"
OUTCSV = OUTDIR / "qam32_raw_label_vs_vacuum_dense_extra.csv"


@dataclass(frozen=True)
class Candidate:
    loss_db: float
    d: float
    source: str


@dataclass(frozen=True)
class CandidateResult:
    loss_db: float
    d: float
    source: str
    vacuum_rate: float
    vacuum_success: float
    vacuum_fidelity: float
    vacuum_useful: int
    raw_rate: float
    raw_success: float
    raw_fidelity: float
    raw_useful: int
    seconds: float


def evaluate_both_receivers(loss_db: float, spacing: float) -> CandidateResult:
    start = time.time()
    eta = 10.0 ** (-loss_db / 10.0)
    local_amps = qam_constellation(32, spacing)
    gram, vac_overlaps = coherent_pair_gram_from_amplitudes(local_amps, eta)
    local_loss = local_loss_coherence_from_amplitudes(local_amps, eta)
    coeffs, _labels, _bell_targets = sparse_measurement_coefficients(32, "bell")

    vacuum_overlaps, vacuum_vec_gram = standard_overlaps_after_vacuum_subtraction(
        coeffs, gram, vac_overlaps
    )
    raw_overlaps, raw_vec_gram = raw_bell_codeword_overlaps(coeffs, gram)

    vacuum_srm_overlaps, vacuum_srm_gram = ykl_square_root_measurement(
        vacuum_overlaps, vacuum_vec_gram
    )
    raw_srm_overlaps, raw_srm_gram = ykl_square_root_measurement(raw_overlaps, raw_vec_gram)

    vacuum_targets = normalized_rows(vacuum_srm_overlaps)
    raw_targets = normalized_rows(raw_srm_overlaps)

    vacuum = evaluate_strategy_factorized_loss(
        vacuum_srm_overlaps,
        vacuum_srm_gram,
        vacuum_targets,
        local_loss,
        32,
        rank_tol=LOSS_RANK_TOL,
    )
    raw = evaluate_strategy_factorized_loss(
        raw_srm_overlaps,
        raw_srm_gram,
        raw_targets,
        local_loss,
        32,
        rank_tol=LOSS_RANK_TOL,
    )
    return CandidateResult(
        loss_db=loss_db,
        d=spacing,
        source="",
        vacuum_rate=float(vacuum.rate),
        vacuum_success=float(vacuum.success_probability),
        vacuum_fidelity=float(vacuum.average_fidelity),
        vacuum_useful=int(vacuum.useful_outcomes),
        raw_rate=float(raw.rate),
        raw_success=float(raw.success_probability),
        raw_fidelity=float(raw.average_fidelity),
        raw_useful=int(raw.useful_outcomes),
        seconds=time.time() - start,
    )


def interp_from_coarse(loss: float, column: str) -> float:
    coarse = pd.read_csv(COARSE_SOURCE).sort_values("loss_db")
    usable = coarse[(coarse["loss_db"] > 0.0) & (coarse["loss_db"] <= 0.15)]
    return float(np.interp(loss, usable["loss_db"], usable[column]))


def interp_branch_seed(loss: float, branch: str) -> float:
    branches = pd.read_csv(BRANCH_SOURCE)
    branches = branches[(branches["M"].astype(int) == 32) & (branches["branch"] == branch)]
    branches = branches.sort_values("loss_db")
    return float(np.interp(loss, branches["loss_db"], branches["spacing_d"]))


def target_losses() -> list[float]:
    low = [0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09]
    transition = [0.942, 0.944, 0.946, 0.948, 0.952, 0.955, 0.96, 0.97, 0.98, 0.99]
    return low + transition


def candidates_for_loss(loss: float) -> list[Candidate]:
    out: dict[float, str] = {}
    if loss < 0.2:
        seed = interp_from_coarse(loss, "raw_d")
        for mult in (0.95, 1.0, 1.05):
            out[round(seed * mult, 10)] = "low_loss_local"
        if loss <= 0.02:
            for d in (2.2, 2.6):
                out[round(d, 10)] = "low_loss_extra_high_d"
    else:
        high_seed = interp_branch_seed(loss, "high_d")
        low_seed = interp_branch_seed(loss, "low_d")
        for seed, label in ((high_seed, "transition_high_d"), (low_seed, "transition_low_d")):
            out[round(seed, 10)] = label
    return [Candidate(loss, d, source) for d, source in sorted(out.items()) if d > 0]


def read_completed() -> pd.DataFrame:
    if not OUTCSV.exists():
        return pd.DataFrame()
    return pd.read_csv(OUTCSV)


def write_rows(rows: list[dict[str, object]]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "M",
        "loss_db",
        "vacuum_rate",
        "raw_rate",
        "raw_minus_vacuum",
        "vacuum_d",
        "raw_d",
        "vacuum_candidate_source",
        "raw_candidate_source",
        "vacuum_candidate_seconds",
        "raw_candidate_seconds",
        "note",
    ]
    with OUTCSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: float(item["loss_db"])):
            writer.writerow(row)


def evaluate_candidate(candidate: Candidate) -> CandidateResult:
    result = evaluate_both_receivers(candidate.loss_db, candidate.d)
    return CandidateResult(
        loss_db=result.loss_db,
        d=result.d,
        source=candidate.source,
        vacuum_rate=result.vacuum_rate,
        vacuum_success=result.vacuum_success,
        vacuum_fidelity=result.vacuum_fidelity,
        vacuum_useful=result.vacuum_useful,
        raw_rate=result.raw_rate,
        raw_success=result.raw_success,
        raw_fidelity=result.raw_fidelity,
        raw_useful=result.raw_useful,
        seconds=result.seconds,
    )


def row_from_results(loss: float, results: list[CandidateResult]) -> dict[str, object]:
    vacuum = max(results, key=lambda item: item.vacuum_rate)
    raw = max(results, key=lambda item: item.raw_rate)
    return {
        "M": 32,
        "loss_db": f"{loss:.12g}",
        "vacuum_rate": f"{vacuum.vacuum_rate:.12g}",
        "raw_rate": f"{raw.raw_rate:.12g}",
        "raw_minus_vacuum": f"{raw.raw_rate - vacuum.vacuum_rate:.12g}",
        "vacuum_d": f"{vacuum.d:.12g}",
        "raw_d": f"{raw.d:.12g}",
        "vacuum_candidate_source": vacuum.source,
        "raw_candidate_source": raw.source,
        "vacuum_candidate_seconds": f"{vacuum.seconds:.3f}",
        "raw_candidate_seconds": f"{raw.seconds:.3f}",
        "note": "dense extra 32-QAM local branch candidate scan",
    }


def main() -> None:
    completed = read_completed()
    rows = completed.to_dict("records") if not completed.empty else []
    done = set(round(float(loss), 12) for loss in completed.get("loss_db", []))

    for loss in target_losses():
        key = round(loss, 12)
        if key in done:
            print(f"skip loss={loss:g}")
            continue
        candidates = candidates_for_loss(loss)
        results: list[CandidateResult] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(evaluate_candidate, candidate) for candidate in candidates]
            for future in as_completed(futures):
                results.append(future.result())
        row = row_from_results(loss, results)
        rows.append(row)
        write_rows(rows)
        print(
            f"loss={loss:g}: vacuum={float(row['vacuum_rate']):.8g} "
            f"d={float(row['vacuum_d']):.6g}; raw={float(row['raw_rate']):.8g} "
            f"d={float(row['raw_d']):.6g}; candidates={len(candidates)}"
        )

    write_rows(rows)
    print(OUTCSV)


if __name__ == "__main__":
    main()
