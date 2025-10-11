#!/usr/bin/env python3
"""Convert Gaia VOTable to CSV with heliocentric Galactic coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from coordinate_transformations.sky_convert import transform_sphere_to_cartesian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Gaia VOTable to CSV and augment it with heliocentric "
            "Galactic Cartesian coordinates plus tangential velocities in the LSR frame."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("laxman_workspace/data/vela1545.vot"),
        help="Path to the input VOTable (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("laxman_workspace/data/vela1545_with_galactic.csv"),
        help="Path to the output CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of rows processed (useful for quick tests).",
    )
    return parser.parse_args()


def load_votable(path: Path) -> pd.DataFrame:
    """Load VOTable into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Input VOTable not found: {path}")
    table = Table.read(path)
    return table.to_pandas()


def compute_galactic_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute heliocentric Galactic coordinates and velocities."""
    required = ["ra", "dec", "parallax", "pmra", "pmdec"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in VOTable: {missing}")

    valid = df[required].notna().all(axis=1) & (df["parallax"] > 0)
    if not valid.any():
        raise ValueError("No rows with complete astrometry and positive parallax.")

    coords_df = transform_sphere_to_cartesian(
        ra=df.loc[valid, "ra"].values,
        dec=df.loc[valid, "dec"].values,
        parallax=df.loc[valid, "parallax"].values,
        pmra=df.loc[valid, "pmra"].values,
        pmdec=df.loc[valid, "pmdec"].values,
    )
    coords_df.index = df.index[valid]
    for column in coords_df.columns:
        df[column] = coords_df[column]
    return df


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file {args.output} exists. Use --overwrite to replace it."
        )

    df = load_votable(args.input)
    if args.limit is not None:
        df = df.iloc[: args.limit].copy()
    df = compute_galactic_coordinates(df)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
