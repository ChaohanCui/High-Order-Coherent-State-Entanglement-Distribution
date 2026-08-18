#!/usr/bin/env python3
"""Recompute representative manuscript data points from the simulation source.

This is a deliberately small audit script.  It does not rerun the full scans,
but it does import the release copy of the physics code and recompute selected
sample points directly from the saved physical parameters.  The goal is to make
sure the curated CSV/NPZ files are tied to unambiguous source-code paths.

Most checks should pass at roundoff-level tolerance.  The optimized 32-outcome
POVM entry is handled separately because the saved result is a nonconvex
best-found matrix.  The script verifies its metadata and POVM completeness, and
then reports whether the current copied objective reproduces the saved rate.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RELEASE_ROOT / "source_code" / "src"
DATA_ROOT = RELEASE_ROOT / "data" / "raw"
DEFAULT_OUTDIR = RELEASE_ROOT / "outputs" / "validation"

# Put the public release source tree first.  The module-path checks below make
# sure imports are not accidentally borrowed from the larger working directory.
sys.path.insert(0, str(SOURCE_ROOT))

from compare_schmidt_bell_povm_qam import normalized_rows, raw_bell_codeword_overlaps
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
from qam_reflection_source_loss_hashing import evaluate_reflection_srm

import compare_schmidt_bell_povm_qam
import mpsk_ghz_hashing
import optimize_qam4_general_povm
import qam_hashing
import qam_reflection_source_loss_hashing
import qam_source_loss_hashing


@dataclass(frozen=True)
class CheckRecord:
    """One row in the reproducibility spot-check report."""

    category: str
    check: str
    expected: float | str
    actual: float | str
    abs_error: float | str
    tolerance: float | str
    status: str
    note: str


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for the spot-check CSV report.",
    )
    return p


def module_location_check(module: ModuleType) -> CheckRecord:
    path = Path(module.__file__).resolve()
    status = "PASS" if path.is_relative_to(SOURCE_ROOT) else "FAIL"
    return CheckRecord(
        category="source_import",
        check=module.__name__,
        expected=str(SOURCE_ROOT),
        actual=str(path),
        abs_error="",
        tolerance="",
        status=status,
        note="module imported from release source tree",
    )


def add_numeric_check(
    rows: list[CheckRecord],
    category: str,
    name: str,
    expected: float,
    actual: float,
    tolerance: float,
    note: str,
) -> None:
    err = abs(float(actual) - float(expected))
    status = "PASS" if err <= tolerance else "FAIL"
    rows.append(
        CheckRecord(
            category=category,
            check=name,
            expected=f"{float(expected):.16g}",
            actual=f"{float(actual):.16g}",
            abs_error=f"{err:.3e}",
            tolerance=f"{tolerance:.3e}",
            status=status,
            note=note,
        )
    )


def one_row(df: pd.DataFrame, mask: pd.Series, description: str) -> pd.Series:
    sub = df[mask]
    if sub.empty:
        raise ValueError(f"Could not find saved row for {description}.")
    if len(sub) > 1:
        raise ValueError(f"Found multiple saved rows for {description}.")
    return sub.iloc[0]


def evaluate_ideal_srm(
    m: int,
    loss_db: float,
    spacing: float,
    receiver: str,
):
    """Evaluate full-state or vacuum-omit SRM for an ideal QAM channel point."""

    eta = 10.0 ** (-loss_db / 10.0)
    local_amps = qam_constellation(m, spacing)
    gram, vac_overlaps = coherent_pair_gram_from_amplitudes(local_amps, eta)
    local_loss = local_loss_coherence_from_amplitudes(local_amps, eta)
    coeffs, _labels, _targets = sparse_measurement_coefficients(m, "bell")

    if receiver == "raw_label_srm":
        overlaps, vec_gram = raw_bell_codeword_overlaps(coeffs, gram)
    elif receiver == "vacuum_omit_srm":
        overlaps, vec_gram = standard_overlaps_after_vacuum_subtraction(
            coeffs, gram, vac_overlaps
        )
    else:
        raise ValueError(f"Unknown receiver: {receiver}")

    # Historical scripts call this helper ykl_square_root_measurement; here it
    # is the square-root measurement applied to the selected input codewords.
    srm_overlaps, srm_gram = ykl_square_root_measurement(overlaps, vec_gram)
    targets = normalized_rows(srm_overlaps)
    return evaluate_strategy_factorized_loss(
        srm_overlaps,
        srm_gram,
        targets,
        local_loss,
        m,
        rank_tol=LOSS_RANK_TOL,
    )


def run_checks() -> list[CheckRecord]:
    rows: list[CheckRecord] = []

    for module in [
        mpsk_ghz_hashing,
        qam_hashing,
        compare_schmidt_bell_povm_qam,
        qam_source_loss_hashing,
        qam_reflection_source_loss_hashing,
        optimize_qam4_general_povm,
    ]:
        rows.append(module_location_check(module))

    ideal = pd.read_csv(DATA_ROOT / "ideal_channel_srm" / "raw_label_vs_vacuum_omit_srm_all_data.csv")
    ideal_m4 = one_row(
        ideal,
        (ideal["M"].astype(int) == 4) & (np.abs(ideal["loss_db"].astype(float) - 0.5) < 1.0e-12),
        "4-QAM ideal loss 0.5 dB",
    )
    for receiver, d_col, rate_col in [
        ("raw_label_srm", "raw_d", "raw_rate"),
        ("vacuum_omit_srm", "vacuum_d", "vacuum_rate"),
    ]:
        result = evaluate_ideal_srm(4, 0.5, float(ideal_m4[d_col]), receiver)
        add_numeric_check(
            rows,
            "ideal_channel_srm",
            f"4-QAM loss=0.5 {receiver} rate at saved d",
            float(ideal_m4[rate_col]),
            result.rate,
            5.0e-10,
            "recomputed from qam_hashing.py and compare_schmidt_bell_povm_qam.py",
        )

    qam32 = pd.read_csv(DATA_ROOT / "ideal_channel_srm" / "qam32_raw_label_vs_vacuum_merged.csv")
    qam32_zero = one_row(
        qam32,
        (qam32["M"].astype(int) == 32) & (np.abs(qam32["loss_db"].astype(float)) < 1.0e-12),
        "32-QAM zero-loss point",
    )
    result_32 = evaluate_ideal_srm(32, 0.0, float(qam32_zero["raw_d"]), "raw_label_srm")
    add_numeric_check(
        rows,
        "ideal_channel_srm",
        "32-QAM zero-loss raw-label SRM rate",
        float(qam32_zero["raw_rate"]),
        result_32.rate,
        1.0e-5,
        "zero-loss 32-QAM check; looser tolerance avoids numerical rank noise",
    )

    sweep = pd.read_csv(DATA_ROOT / "d_sweep" / "raw_label_srm_qam_branch_sweep_points.csv")
    sweep_row = one_row(
        sweep,
        (sweep["M"].astype(int) == 4)
        & (np.abs(sweep["loss_db"].astype(float) - 0.9) < 1.0e-12)
        & (np.abs(sweep["spacing_d"].astype(float) - 1.715) < 1.0e-12),
        "4-QAM d sweep loss 0.9 dB, d=1.715",
    )
    sweep_result = evaluate_ideal_srm(4, 0.9, float(sweep_row["spacing_d"]), "raw_label_srm")
    add_numeric_check(
        rows,
        "fig2b_d_sweep",
        "4-QAM loss=0.9 d=1.715 raw-label SRM rate",
        float(sweep_row["hashing_bound_bits_per_attempt"]),
        sweep_result.rate,
        5.0e-10,
        "recomputed d-sweep sample at fixed d",
    )

    interface = pd.read_csv(
        DATA_ROOT
        / "interface_loss"
        / "reflection_source_raw_label_global_optima_with_ultradense16.csv"
    )
    interface_row = one_row(
        interface,
        (interface["M"].astype(int) == 4)
        & (np.abs(interface["generation_loss_db_per_step"].astype(float) - 0.1) < 1.0e-12),
        "4-QAM interface loss 0.1 dB",
    )
    eta_source = 10.0 ** (-float(interface_row["generation_loss_db_per_step"]) / 10.0)
    eta_channel = 10.0 ** (-float(interface_row["channel_loss_db"]) / 10.0)
    interface_result = evaluate_reflection_srm(
        m=4,
        spacing=float(interface_row["spacing_d"]),
        eta_source=eta_source,
        eta_channel=eta_channel,
        source_loss_db=float(interface_row["generation_loss_db_per_step"]),
        channel_loss_db=float(interface_row["channel_loss_db"]),
        convention="reflection",
        rank_tol=1.0e-8,
        receiver="raw_label_srm",
    )
    add_numeric_check(
        rows,
        "fig3_interface_loss",
        "4-QAM source loss=0.1 dB raw-label SRM rate",
        float(interface_row["hashing_bound_bits_per_attempt"]),
        interface_result.result.rate,
        5.0e-10,
        "recomputed reflection-source/interface-loss sample",
    )

    phase = pd.read_csv(DATA_ROOT / "phase_error" / "interface_0p1db" / "raw_label_phase_error_summary.csv")
    phase_row = one_row(
        phase,
        (phase["M"].astype(int) == 4)
        & (np.abs(phase["phase_error_pi"].astype(float) - 0.115) < 1.0e-12),
        "4-QAM phase-bias point delta_phi=0.115 pi",
    )
    phase_result = evaluate_reflection_srm(
        m=4,
        spacing=float(phase_row["optimized_spacing_d"]),
        eta_source=float(phase_row["eta_source"]),
        eta_channel=float(phase_row["eta_channel"]),
        source_loss_db=float(phase_row["source_loss_db_per_interface"]),
        channel_loss_db=float(phase_row["channel_loss_db_per_arm"]),
        convention="reflection",
        rank_tol=1.0e-8,
        phase_error_rad=float(phase_row["phase_error_rad"]),
        receiver="raw_label_srm",
    )
    add_numeric_check(
        rows,
        "fig3_phase_bias",
        "4-QAM interface=0.1 dB delta_phi=0.115 pi raw-label SRM rate",
        float(phase_row["hashing_bound_bits_per_attempt"]),
        phase_result.result.rate,
        1.0e-9,
        "recomputed phase-bias sample",
    )

    selected = pd.read_csv(DATA_ROOT / "optimized_povm" / "qam4_selected_32outcome_povm_comparison.csv").iloc[0]
    spacing = float(selected["spacing_d"])
    for receiver, expected_col in [
        ("raw_label_srm", "raw_label_srm_rate_at_same_d"),
        ("vacuum_omit_srm", "vacuum_omit_srm_rate_at_same_d"),
    ]:
        result = evaluate_ideal_srm(4, float(selected["loss_db_per_arm"]), spacing, receiver)
        add_numeric_check(
            rows,
            "fig5b_srm_baseline",
            f"4-QAM loss=0.25 d={spacing:.12g} {receiver} rate",
            float(selected[expected_col]),
            result.rate,
            5.0e-10,
            "deterministic SRM baseline used beside the optimized POVM",
        )

    matrix_path = RELEASE_ROOT / str(selected["best_matrix_path_in_release"])
    matrix = np.load(matrix_path)
    matrix_spacing = float(matrix["spacing_d"])
    matrix_previous_rate = float(matrix["previous_ykl_rate"])
    add_numeric_check(
        rows,
        "fig5b_optimized_povm",
        "selected CSV spacing_d matches saved POVM metadata",
        matrix_spacing,
        float(selected["spacing_d"]),
        5.0e-16,
        "selected comparison table uses the saved POVM metadata",
    )
    add_numeric_check(
        rows,
        "fig5b_optimized_povm",
        "NPZ best_rate metadata matches selected comparison CSV",
        float(selected["optimized_32outcome_povm_rate"]),
        float(matrix["best_rate"]),
        5.0e-11,
        "metadata consistency check",
    )
    a = matrix["A"]
    completeness = float(np.linalg.norm(a.conj().T @ a - np.eye(a.shape[1])))
    add_numeric_check(
        rows,
        "fig5b_optimized_povm",
        "saved rank-one POVM row matrix column completeness",
        0.0,
        completeness,
        1.0e-10,
        "checks A^dagger A = I on the optical support",
    )

    # The saved POVM file is the canonical record for this optimized point.
    problem = optimize_qam4_general_povm.build_problem(
        float(selected["loss_db_per_arm"]),
        matrix_spacing,
        matrix_previous_rate,
    )
    current_rate, current_success, current_useful, _grad = optimize_qam4_general_povm.objective_and_gradient(
        a,
        problem,
        gradient=False,
    )
    add_numeric_check(
        rows,
        "fig5b_optimized_povm",
        "current copied optimizer objective evaluated on saved A",
        float(selected["optimized_32outcome_povm_rate"]),
        current_rate,
        5.0e-10,
        (
            "current objective reports success="
            f"{current_success:.12g}, useful_outcomes={current_useful}; "
            "evaluated from saved POVM metadata"
        ),
    )

    return rows


def main() -> None:
    args = parser().parse_args()
    rows = run_checks()
    args.outdir.mkdir(parents=True, exist_ok=True)
    report_path = args.outdir / "spot_check_simulation_samples.csv"
    pd.DataFrame([asdict(row) for row in rows]).to_csv(report_path, index=False)

    for row in rows:
        print(
            f"[{row.status}] {row.category}: {row.check} "
            f"expected={row.expected} actual={row.actual} err={row.abs_error}"
        )
        if row.status == "FAIL":
            print(f"       {row.note}")

    print(f"Wrote {report_path}")
    failures = [row for row in rows if row.status == "FAIL"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
