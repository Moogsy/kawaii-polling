#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B: Join ratings with per-image geometry features and fit mixed-effects models.

By default, uses *external* ratings only (Model != RaterID). You can switch to include
self-ratings via a flag. The model follows the paper's formula:

Score ~ C(Category) + COG_Y + COG_X + Elbow_Min_Angle + Knee_Min_Angle
        + R_Arm_Ratio + L_Arm_Ratio + Head_Tilt + Head_Rot
        + (1 | RaterID) + (1 | Model) + (1 | ImageID)

Implemented via statsmodels MixedLM:
- 'groups' = RaterID (random intercept)
- Variance components for Model and ImageID via vc_formula

Usage
-----
python 02_fit_mixedlm.py \
  --ratings concat_ratings.csv \
  --features features_pose.csv \
  --out-prefix results/mixedlm \
  [--include-self] \
  [--image-key image_path] [--merge-on image_path] \
  [--standardize]

Merging logic
-------------
- Preferred: merge on a unique key present in both files (default: image_path).
- Fallbacks are provided if you specify --merge-on Category,Model,ImageID etc.

Outputs
-------
- <out-prefix>_model_<DIM>.txt   : statsmodels textual summary
- <out-prefix>_coefs_<DIM>.csv   : fixed effects table
- <out-prefix>_varcomps_<DIM>.csv: random effects variance components
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FEATURES = [
    'COG_Y', 'COG_X',
    'Elbow_Min_Angle', 'Knee_Min_Angle',
    'R_Arm_Ratio', 'L_Arm_Ratio',
    'Head_Tilt', 'Head_Rot'
]
CAT_COL = 'Category'
ID_IMAGE = 'ImageID'
ID_MODEl = 'Model'
ID_RATER = 'RaterID'

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def standardize_inplace(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            mu = df[c].mean()
            sd = df[c].std(ddof=0)
            if sd > 0:
                df[c] = (df[c] - mu) / sd

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True, help="concat_ratings.csv")
    ap.add_argument("--features", required=True, help="features_pose.csv from Phase A")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for results")
    ap.add_argument("--include-self", action="store_true", help="Include self-ratings (default: external-only)")
    ap.add_argument("--merge-on", default="image_path",
                    help="Column (or comma-separated columns) to merge on (must exist in both)")
    ap.add_argument("--standardize", action="store_true", help="Z-standardize continuous predictors")
    args = ap.parse_args()

    ratings = pd.read_csv(args.ratings)
    feats   = pd.read_csv(args.features)

    # Normalize identifiers
    for col in ["Model", "RaterID", "Category"]:
        if col in ratings.columns:
            ratings[col] = ratings[col].astype(str).str.strip()
    for col in ["Model", "Category"]:
        if col in feats.columns:
            feats[col] = feats[col].astype(str).str.strip()

    # Filter external ratings by default
    if not args.include_self:
        if "Model" in ratings.columns and "RaterID" in ratings.columns:
            m = ratings["Model"].astype(str).str.lower().str.strip()
            r = ratings["RaterID"].astype(str).str.lower().str.strip()
            ratings = ratings[m != r]
            logging.info(f"External ratings only: {len(ratings)} rows after filtering")
        else:
            logging.warning("Model/RaterID not found; cannot filter self-ratings.")

    # Create merge key(s)
    merge_keys = [k.strip() for k in args.merge_on.split(",")]
    missing_left  = [k for k in merge_keys if k not in ratings.columns]
    missing_right = [k for k in merge_keys if k not in feats.columns]
    if missing_left or missing_right:
        raise ValueError(f"Merge keys missing. Left missing: {missing_left}, Right missing: {missing_right}")

    df = ratings.merge(feats, on=merge_keys, how="inner")
    logging.info(f"Merged rows: {len(df)}")

    # Guard rails
    needed = [CAT_COL, ID_MODEl, ID_RATER, "Rating", "Score"] + FEATURES
    for c in needed:
        if c not in df.columns:
            logging.warning(f"Column '{c}' not found after merge.")
    # Use ImageID for variance component if present; else derive from merge key
    if ID_IMAGE not in df.columns:
        # Derive a pseudo ImageID from merge keys
        df[ID_IMAGE] = df[merge_keys].astype(str).agg("|".join, axis=1)

    # Optionally standardize continuous predictors
    if args.standardize:
        standardize_inplace(df, FEATURES)

    # Fit per-dimension models (e.g., 'Kawaii', 'Warmth', 'Expressiveness', ...)
    dims = sorted(df["Rating"].dropna().unique().tolist())
    out_dir = Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for dim in dims:
        sub = df[df["Rating"] == dim].copy()
        sub = sub.dropna(subset=["Score"])
        # MixedLM: groups = RaterID; variance components for Model and ImageID
        # Fixed part: Category as categorical + geometry features
        fixed_terms = "C(Category) + " + " + ".join([f for f in FEATURES if f in sub.columns])
        formula = f"Score ~ {fixed_terms}"

        # Build vc_formula dict (variance components) using 0 + C(factor)
        vc = {}
        if ID_MODEl in sub.columns:
            vc["Model"] = "0 + C(Model)"
        if ID_IMAGE in sub.columns:
            vc["ImageID"] = "0 + C(ImageID)"

        logging.info(f"[{dim}] Fitting MixedLM with formula: {formula}")
        logging.info(f"[{dim}] Variance components: {list(vc.keys())}")

        try:
            model = smf.mixedlm(formula, data=sub, groups=sub[ID_RATER], vc_formula=vc, re_formula="1")
            res = model.fit(method="lbfgs", maxiter=200, disp=False)
        except Exception as e:
            logging.exception(f"[{dim}] MixedLM failed; trying Powell optimizer.")
            model = smf.mixedlm(formula, data=sub, groups=sub[ID_RATER], vc_formula=vc, re_formula="1")
            res = model.fit(method="powell", maxiter=300, disp=False)

        # Save textual summary
        summ_path = f"{args.out_prefix}_model_{dim}.txt"
        with open(summ_path, "w") as f:
            f.write(res.summary().as_text())

        # Fixed effects
        coefs = pd.DataFrame({
            "term": res.params.index,
            "estimate": res.params.values,
            "se": res.bse.values
        })
        coefs.to_csv(f"{args.out_prefix}_coefs_{dim}.csv", index=False)

        # Variance components (random)
        vc_out = []
        # Random intercept (RaterID group variance)
        if "Group Var" in res.cov_re:
            vc_out.append({"component": "RaterID", "var": float(res.cov_re.iloc[0,0])})
        elif hasattr(res, "random_effects"):
            # fallback best-effort: variance of BLUPs
            blups = pd.DataFrame(res.random_effects).T.iloc[:,0]
            vc_out.append({"component": "RaterID", "var": float(blups.var(ddof=1))})
        # Additional variance components
        if hasattr(res, "vcomp") and res.vcomp is not None:
            for name, val in zip(res.model.vc_names, res.vcomp):
                vc_out.append({"component": name, "var": float(val)})

        pd.DataFrame(vc_out).to_csv(f"{args.out_prefix}_varcomps_{dim}.csv", index=False)

        logging.info(f"[{dim}] Done. Saved: {summ_path}")

if __name__ == "__main__":
    main()

