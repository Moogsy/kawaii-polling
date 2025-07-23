#!/usr/bin/env python3
# export_agg_scores.py

import logging
from pathlib import Path
import pandas as pd

# --- Configuration Logging ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# --- Load & clean data ---
def load_df_from_csv(csv_dir: Path) -> pd.DataFrame:
    logging.info(f"Loading CSV files from {csv_dir}")
    dfs = []
    for path in csv_dir.glob("*.csv"):
        logging.info(f"Reading {path.name}")
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
    if not dfs:
        logging.error(f"No CSV files found in {csv_dir}")
        return pd.DataFrame()
    concatenated = pd.concat(dfs, ignore_index=True)
    logging.info(f"Total records loaded: {len(concatenated)}")
    return concatenated


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    orig = len(df)
    df_clean = df[df['RaterID'] != df['Model']].reset_index(drop=True)
    removed = orig - len(df_clean)
    logging.info(f"Removed {removed} self-ratings; {len(df_clean)} records remain")
    return df_clean # type: ignore

# --- Aggregate mean kawaii scores ---
def aggregate_mean_scores(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df
        .groupby(['Category', 'Model'])['Score']
        .mean()
        .reset_index()
        .rename(columns={'Score': 'kawaii_mean', 'Model': 'ModelID'})
    )
    logging.info(f"Aggregated into {len(agg)} unique (Category, Model) pairs")
    return agg

# --- Construct image paths ---
def construct_image_paths(agg: pd.DataFrame, static_dir: Path) -> pd.DataFrame:
    agg['image_path'] = agg.apply(
        lambda r: static_dir / r['Category'] / f"{r['ModelID']}.jpg",
        axis=1
    )
    logging.info("Constructed image paths for all records")
    return agg

# --- Main execution ---
def main():
    setup_logging()
    logging.info("=== Export Aggregated Scores ===")

    # Hardcoded paths
    project_root = Path(__file__).resolve().parent.parent
    ratings_dir  = project_root / "Ratings"
    static_dir   = project_root / "static"
    output_csv   = project_root / "AI" / "agg_phase2.csv"

    # 1. Load and clean
    df_raw   = load_df_from_csv(ratings_dir)
    if df_raw.empty:
        logging.error("No data loaded. Exiting.")
        return
    df_clean = clean_df(df_raw)

    # 2. Aggregate
    df_agg = aggregate_mean_scores(df_clean)

    # 3. Image paths
    df_agg = construct_image_paths(df_agg, static_dir)

    # 4. Save to CSV
    df_agg.to_csv(output_csv, index=False)
    logging.info(f"Aggregated CSV saved to {output_csv}")

if __name__ == '__main__':
    main()

