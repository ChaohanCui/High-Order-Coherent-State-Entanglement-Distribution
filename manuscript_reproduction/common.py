"""Shared helpers for manuscript data extraction scripts.

The release folder is designed to be run directly from a GitHub checkout.  The
helpers below keep all paths relative to the release root and give each script
the same lightweight command-line style.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = RELEASE_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = RELEASE_ROOT / "outputs" / "subfigure_data"


def parser(description: str) -> argparse.ArgumentParser:
    """Return a standard parser used by all figure-data scripts."""

    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing curated raw data files.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where subfigure-ready CSV files are written.",
    )
    return p


def read_csv(data_root: Path, relative_path: str) -> pd.DataFrame:
    """Read a release data CSV and fail with a clear path if it is missing."""

    path = data_root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing curated data file: {path}")
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, outdir: Path, filename: str) -> Path:
    """Write a DataFrame into the standard output directory."""

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    df.to_csv(path, index=False)
    return path


def add_loss_conventions(df: pd.DataFrame, loss_col: str = "loss_db") -> pd.DataFrame:
    """Add explicit loss-convention columns.

    Most simulation scripts use a per-arm channel loss, i.e. Alice-to-Charlie
    and Bob-to-Charlie are each ``loss_db``.  Some manuscript labels use the
    end-to-end Alice-Bob loss, which is twice the per-arm dB value for a
    symmetric midpoint geometry.  Keeping both columns avoids ambiguity.
    """

    out = df.copy()
    out["loss_db_per_arm_code"] = out[loss_col].astype(float)
    out["end_to_end_loss_db_midpoint"] = 2.0 * out["loss_db_per_arm_code"]
    return out


def ordered_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return known columns first and leave any extra diagnostic columns after."""

    front = [col for col in columns if col in df.columns]
    rest = [col for col in df.columns if col not in front]
    return df[front + rest]

