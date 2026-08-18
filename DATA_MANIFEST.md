# Data Manifest

This manifest lists the curated data files included in this release and the
manuscript subfigures they support.

## Ideal Channel Loss, Full-State SRM

- `data/raw/ideal_channel_srm/raw_label_vs_vacuum_omit_srm_all_data.csv`
  - M = 2, 4, 8, 16 QAM.
  - Columns include full-state/raw-label SRM rate and d, vacuum-omit SRM rate
    and d, and the difference.
  - Used by Fig. 1(c), Fig. 2(a), and End Matter Fig. 5(a).
- `data/raw/ideal_channel_srm/qam32_raw_label_vs_vacuum_merged.csv`
  - M = 32 QAM.
  - Merges seeded, dense-extra, and low-loss SciPy-refined rows.
  - Used by Fig. 1(c), Fig. 2(a), and End Matter Fig. 5(a).

## Fixed-Loss d Sweeps

- `data/raw/d_sweep/raw_label_srm_qam_branch_sweep_points.csv`
  - Full-state SRM hashing bound versus spacing d.
  - M = 4, 8, 16; channel losses 0.9 and 0.95 dB per arm.
  - Used by Fig. 2(b).
- `data/raw/d_sweep/raw_label_srm_qam_branch_sweep_local_maxima.csv`
  - Local maxima extracted from the d-sweep curves.
  - Used to identify branch switches in Fig. 2(b).

## Interface Loss

- `data/raw/interface_loss/reflection_source_raw_label_global_optima_with_ultradense16.csv`
  - Full-state SRM global optimum versus interface loss.
  - M = 2, 4, 8, 16; per-arm channel loss = 0.25 dB.
  - Used by Fig. 3(a,b).
- `data/raw/interface_loss/reflection_source_raw_label_local_branches.csv`
  - Local branch details for the raw-label interface-loss scan.
- `data/raw/interface_loss/reflection_source_16qam_raw_label_ultradense_transition_global.csv`
  - Ultra-dense 16-QAM transition refinement.
- `data/raw/interface_loss/reflection_source_raw_vs_vacuum_interpolated_with_ultradense16.csv`
  - Comparison data used for End Matter raw-vs-vacuum interface checks.

## Phase Bias

- `data/raw/phase_error/interface_0p1db/raw_label_phase_error_summary.csv`
  - Full-state SRM phase-bias sweep for 0.1 dB interface loss.
  - M = 2, 4, 8; phase error from -0.3 pi to +0.3 pi.
- `data/raw/phase_error/interface_0p2db/raw_label_phase_error_summary.csv`
  - Full-state SRM phase-bias sweep for 0.2 dB interface loss.
  - M = 2, 4, 8; phase error from -0.3 pi to +0.3 pi.
- `data/raw/phase_error/interface_0p1db/reflection_source_phase_error_raw_vs_vacuum_comparison.csv`
- `data/raw/phase_error/interface_0p2db/reflection_source_phase_error_raw_vs_vacuum_comparison.csv`
  - Full-state versus vacuum-omit comparison tables for the same phase sweeps.

## Optimized 32-Outcome POVM

- `data/raw/optimized_povm/qam4_selected_32outcome_povm_comparison.csv`
  - Selected 4-QAM, 0.25 dB per-arm loss, 32-outcome POVM result and SRM
    baselines.
- `data/raw/optimized_povm/best_by_d_scale.csv`
  - Best-found POVM rate versus d-scale.
- `data/raw/optimized_povm/all_d_scale_trials.csv`
  - All multi-start d-scale trial summaries.
- `data/raw/optimized_povm/best_M4_loss_0.25_scale_0p93_outcomes_32.npz`
  - Complex rank-one POVM row matrix for the selected best-found result.
  - Metadata and POVM completeness are checked by
    `scripts/spot_check_simulation_samples.py`.
  - The selected comparison script loads this file as the canonical metadata
    source for the optimized-POVM point.
- `data/raw/optimized_povm/*bell_fidelity_matrix.csv`
- `data/raw/optimized_povm/*outcome_fingerprints.csv`
  - Fingerprint/visualization data for the SRM and optimized POVM outcomes.

## External Baselines

- `data/raw/external_baselines/README.md`
  - Placeholder documentation for PLOB, CTW, single-photon, and Hex-GKP curves
    if the final Fig. 1(c) plotting script is published.
