#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Primary statistical model of the paper (§4.4 / Equation 1 / Tables 11–12).
Implements the mixed-effects model:

    Score ~ C(Category) + COG_Y + COG_X + Elbow_Min_Angle + Knee_Min_Angle
            + R_Arm_Ratio + L_Arm_Ratio + Head_Tilt + Head_Rot
            + (1 | RaterID) + (1 | Model) + (1 | ImageID)

COG_Y (vertical center of gravity) is the only significant geometric predictor
(β = 0.274, p = 0.007).  Outputs fixed-effect tables and summaries to
mixedlm_outputs/.  Computes Nakagawa & Schielzeth marginal/conditional R².

Replaces RF with LMM using the same MediaPipe keypoints & geometric features.

Inputs
------
- concat_ratings.csv (Phase 2 ratings): columns like
    ['Category','Model','RaterID','Rating','Score', ...]
  We will keep external ratings only: Model != RaterID.
  We will model one dimension at a time (e.g., Rating == 'Kawaii').

- Images under a predictable layout, or a column 'image_path' mapping each row
  to the corresponding (blurred) pose image.

Main model (per dimension)
--------------------------
Score ~ C(Category) + COG_Y + COG_X + Elbow_Min_Angle + Knee_Min_Angle
        + R_Arm_Ratio + L_Arm_Ratio + Head_Tilt + Head_Rot
        + (1 | RaterID) + (1 | Model) + (1 | ImageID)

Optionally add random slopes for COG_Y by RaterID if convergence allows.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import statsmodels.api as sm
from patsy import dmatrices

# ============================================================================
# Configuration
# ============================================================================

# Name of the CSV with ratings (external observers, Phase 2)
RATINGS_CSV = "../Analysis/concat_ratings.csv"   # or "agg_phase2.csv" if that’s your aggregate
IMAGE_ROOT  = Path("../static")                  # adjust to your layout
IMAGE_COL   = "image_path"                       # set or constructed below

# Which rating dimensions to model
DIMENSIONS_TO_MODEL = ["Kawaii", "Warmth", "Expressiveness"]

# If your rows don’t have image paths, we’ll build them from Category/ModelID
# Customize this lambda if your naming differs.
BUILD_IMAGE_PATH = lambda row: (
    IMAGE_ROOT / row["Category"] / f"blurred_{row.get('ModelID', row['Model'])}.png"
)

# Set to "COG_Y" (or another geometric feature) to enable a random slope by RaterID.
# Set to None to keep a simple random intercept for raters.
RANDOM_SLOPE_FOR = None  # e.g., "COG_Y"

# ============================================================================
# Logging & Utils
# ============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def zscore(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        mu = out[c].mean()
        sd = out[c].std(ddof=1)
        if sd and sd > 1e-12:
            out[c] = (out[c] - mu) / sd
    return out

# ============================================================================
# MediaPipe Pose Extraction & Geometry
# ============================================================================

class PoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)

    def extract(self, image_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        img = cv2.imread(str(image_path))
        if img is None:
            logging.warning(f"Cannot read image: {image_path}")
            return None, None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            logging.warning(f"No landmarks: {image_path}")
            return None, None
        pts = np.array([(lm.x, lm.y) for lm in res.pose_landmarks.landmark])
        vis = np.array([lm.visibility for lm in res.pose_landmarks.landmark])
        return pts, vis

def angle_between(p1: np.ndarray, p2: np.ndarray) -> float:
    vec = p1 - p2
    return np.degrees(np.arctan2(vec[1], vec[0]))

def rotate(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s,  c]])
    return pts @ R.T

def pca_orientation(pts: np.ndarray) -> float:
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(axis[1], axis[0]))
    return angle + 180 if angle < 0 else angle

def normalize_pose(pts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    idx = {'LSHO':11, 'RSHO':12, 'LHIP':23, 'RHIP':24}
    mid_hip = (pts[idx['LHIP']] + pts[idx['RHIP']]) / 2
    mid_sho = (pts[idx['LSHO']] + pts[idx['RSHO']]) / 2
    centered = pts - mid_hip
    torso_len = np.linalg.norm(mid_sho - mid_hip)
    scale = torso_len if torso_len > 1e-6 else 1.0
    scaled = centered / scale

    def line_angle(a, b, va, vb):
        if va > 0 and vb > 0:
            return angle_between(a, b), (va + vb) / 2
        return None, 0.0

    sa, sw = line_angle(scaled[idx['LSHO']], scaled[idx['RSHO']], vis[idx['LSHO']], vis[idx['RSHO']])
    ha, hw = line_angle(scaled[idx['LHIP']], scaled[idx['RHIP']], vis[idx['LHIP']], vis[idx['RHIP']])
    sa, sw = (sa or 0.0, sw)
    ha, hw = (ha or 0.0, hw)
    W = sw + hw
    target = (sa*sw + ha*hw)/W if W > 0 else pca_orientation(scaled)
    norm = rotate(scaled, -target)

    hip_center = (norm[idx['LHIP']] + norm[idx['RHIP']]) / 2
    sho_center = (norm[idx['LSHO']] + norm[idx['RSHO']]) / 2
    if (sho_center - hip_center)[1] < 0:
        norm[:,1] *= -1
    return norm

def joint_angle(a, b, c) -> float:
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_v = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0)))

FEATURE_NAMES = [
    'Elbow_Min_Angle', 'Elbow_Delta_Angle',
    'Knee_Min_Angle', 'Knee_Delta_Angle',
    'R_Arm_Ratio', 'L_Arm_Ratio',
    'COG_X', 'COG_Y',
    'Head_Tilt', 'Head_Rot',
]

def compute_geom_features(pts: np.ndarray) -> np.ndarray:
    idx = {
        'RSHO': 12, 'RELB': 14, 'RWRA': 16,
        'LSHO': 11, 'LELB': 13, 'LWRA': 15,
        'RHIP': 24, 'RKNE': 26, 'RANK': 28,
        'LHIP': 23, 'LKNE': 25, 'LANK': 27,
        'LEAR': 7,  'REAR': 8,  'NOSE': 0
    }
    dist = lambda u, v: np.linalg.norm(u - v)

    elbows = [
        joint_angle(pts[idx['RSHO']], pts[idx['RELB']], pts[idx['RWRA']]),
        joint_angle(pts[idx['LSHO']], pts[idx['LELB']], pts[idx['LWRA']]),
    ]
    knees = [
        joint_angle(pts[idx['RHIP']], pts[idx['RKNE']], pts[idx['RANK']]),
        joint_angle(pts[idx['LHIP']], pts[idx['LKNE']], pts[idx['LANK']]),
    ]
    elbow_min  = min(elbows)
    elbow_delta = max(elbows) - elbow_min
    knee_min   = min(knees)
    knee_delta  = max(knees) - knee_min

    r_arm_ratio = dist(pts[idx['RSHO']], pts[idx['RELB']]) / (dist(pts[idx['RSHO']], pts[idx['RHIP']]) + 1e-8)
    l_arm_ratio = dist(pts[idx['LSHO']], pts[idx['LELB']]) / (dist(pts[idx['LSHO']], pts[idx['LHIP']]) + 1e-8)

    cog_x, cog_y = pts.mean(axis=0).tolist()

    head_tilt = angle_between(pts[idx['LEAR']], pts[idx['REAR']])
    head_rot  = dist(pts[idx['NOSE']], pts[idx['LSHO']]) / (dist(pts[idx['NOSE']], pts[idx['RSHO']]) + 1e-8)

    feats = [
        elbow_min, elbow_delta,
        knee_min,  knee_delta,
        r_arm_ratio, l_arm_ratio,
        cog_x, cog_y,
        head_tilt, head_rot
    ]
    arr = np.array(feats, dtype=float)
    assert arr.shape[0] == len(FEATURE_NAMES)
    return arr

# ============================================================================
# Data assembly
# ============================================================================

def attach_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if IMAGE_COL not in out.columns or out[IMAGE_COL].isna().any():
        out[IMAGE_COL] = out.apply(BUILD_IMAGE_PATH, axis=1).astype(str)
    return out

def build_features_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute geometry for each unique image; join back to each rating row."""
    extr = PoseExtractor()
    df = df.copy()
    df["ImageID"] = df[IMAGE_COL].astype(str)
    uniq = df.drop_duplicates("ImageID")[[ "ImageID", IMAGE_COL, "Category", "Model" ]]

    rows = []
    for _, r in uniq.iterrows():
        p = Path(r[IMAGE_COL])
        pts, vis = extr.extract(p)
        if pts is None:
            continue
        norm = normalize_pose(pts, vis)
        feats = compute_geom_features(norm)
        rec = {"ImageID": r["ImageID"], "Category": r["Category"], "Model": r["Model"]}
        rec.update({FEATURE_NAMES[i]: feats[i] for i in range(len(FEATURE_NAMES))})
        rows.append(rec)

    feats_df = pd.DataFrame(rows)
    return pd.merge(df, feats_df, on=["ImageID", "Category", "Model"], how="inner")

# ============================================================================
# Mixed-effects modelling
# ============================================================================

def nakagawa_r2_mixed(model, result, y, X_fe, Z_dict):
    """
    Compute Nakagawa & Schielzeth (2013) marginal/conditional R^2.
    - model: fitted MixedLM
    - result: fit result
    - y: response vector
    - X_fe: fixed-effects design matrix
    - Z_dict: dict of random-effects design matrices (unused here)
    """
    # Fixed-effects variance
    y_hat_fe = X_fe @ result.fe_params.values
    var_fix = np.var(y_hat_fe, ddof=1)

    # Random effects variance: group RE (rater) + variance components (model, image)
    var_rand = 0.0
    if result.cov_re is not None and result.cov_re.size > 0:
        # random-intercept variance for groups
        var_rand += float(result.cov_re[0, 0])
    if hasattr(result, "vcomp") and result.vcomp is not None:
        var_rand += float(np.sum(result.vcomp))

    # Residual variance
    var_resid = float(result.scale)

    denom = (var_fix + var_rand + var_resid) if (var_fix + var_rand + var_resid) > 0 else np.nan
    r2_marg = var_fix / denom
    r2_cond = (var_fix + var_rand) / denom
    return float(r2_marg), float(r2_cond)

def fit_mixed_for_dimension(df: pd.DataFrame, dim: str):
    # Filter to the target rating dimension (case-insensitive)
    sub = df[df["Rating"].str.lower() == dim.lower()].copy()
    if sub.empty:
        logging.warning(f"No rows for dimension={dim}")
        return None

    # Keep only external ratings: Model != RaterID (normalized)
    sub["model_norm"] = sub["Model"].astype(str).str.strip().str.lower()
    sub["rater_norm"] = sub["RaterID"].astype(str).str.strip().str.lower()
    sub = sub[sub["model_norm"] != sub["rater_norm"]].copy()

    # Ensure IDs are strings
    sub["ImageID"] = sub["ImageID"].astype(str)
    sub["Model"] = sub["Model"].astype(str)
    sub["RaterID"] = sub["RaterID"].astype(str)

    # Fixed-effects formula: Category + geometry
    geom_cols = FEATURE_NAMES.copy()
    sub = zscore(sub, geom_cols)

    fe_terms = " + ".join(["C(Category)"] + geom_cols)
    formula = f"Score ~ {fe_terms}"

    # Variance components for Model and Image (random intercepts as VCs)
    vc = {
        "Model": "0 + C(Model)",
        "Image": "0 + C(ImageID)"
    }

    # Random structure for raters: intercept (+ optional slope)
    if RANDOM_SLOPE_FOR is not None and RANDOM_SLOPE_FOR in geom_cols:
        re_formula = f"1 + {RANDOM_SLOPE_FOR}"
    else:
        re_formula = "1"

    # Use from_formula so vc_formula is honored
    md = sm.MixedLM.from_formula(
        formula=formula,
        data=sub,
        groups="RaterID",
        re_formula=re_formula,
        vc_formula=vc
    )

    # Fit with fallbacks for tough convergence cases
    try:
        res = md.fit(reml=True, method="lbfgs", maxiter=1000, disp=False)
    except Exception as e:
        logging.warning(f"LBFGS (REML) failed: {e}; retrying with ML...")
        try:
            res = md.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
        except Exception as e2:
            logging.warning(f"LBFGS (ML) failed: {e2}; retrying with Powell...")
            res = md.fit(reml=True, method="powell", maxiter=1000, disp=False)

    # Compute Nakagawa R^2 using the FE design matrix
    y, X = dmatrices(formula, sub, return_type="dataframe")
    r2_marg, r2_cond = nakagawa_r2_mixed(md, res, y["Score"].values, X.values, None)

    # Fixed effects table with aligned SEs
    fe = res.fe_params.to_frame("coef")
    # res.bse includes both FE and other params; align by index
    fe["se"] = res.bse.loc[fe.index]

    out = {
        "dimension": dim,
        "n_rows": int(len(sub)),
        "n_models": int(sub["Model"].nunique()),
        "n_raters": int(sub["RaterID"].nunique()),
        "n_images": int(sub["ImageID"].nunique()),
        "r2_marginal": float(r2_marg),
        "r2_conditional": float(r2_cond),
        "fe_params": fe,
        "vc_random": getattr(res, "vcomp", None),
        "summary": res.summary().as_text(),
    }
    return out

# ============================================================================
# Main
# ============================================================================

def main():
    setup_logging()
    logging.info("=== Mixed-effects pose geometry pipeline ===")

    # Load ratings
    ratings = pd.read_csv(RATINGS_CSV)

    # Ensure we have or can build image paths
    if IMAGE_COL not in ratings.columns or ratings[IMAGE_COL].isna().any():
        ratings = attach_image_paths(ratings)

    # Compute geometry per image and join to ratings
    data = build_features_table(ratings)

    # Make sure expected columns exist
    required = ["Category", "Model", "RaterID", "Rating", "Score", "ImageID"] + FEATURE_NAMES
    miss = [c for c in required if c not in data.columns]
    if miss:
        raise ValueError(f"Missing required columns in assembled data: {miss}")

    # Fit per dimension
    reports = []
    for dim in DIMENSIONS_TO_MODEL:
        logging.info(f"--- Fitting MixedLM for {dim} ---")
        res = fit_mixed_for_dimension(data, dim)
        if res is None:
            continue
        reports.append(res)
        # Pretty print key results
        logging.info(f"{dim}: N={res['n_rows']} rows, models={res['n_models']}, raters={res['n_raters']}, images={res['n_images']}")
        logging.info(f"{dim}: R2_marg={res['r2_marginal']:.3f}, R2_cond={res['r2_conditional']:.3f}")
        logging.info("\nFixed effects (coef ± SE):\n" + res["fe_params"].round(3).to_string())
        if res["vc_random"] is not None:
            logging.info(f"Variance components (random): {np.round(res['vc_random'], 4)}")
        logging.info("\n" + res["summary"])

    # Save out CSVs of fixed effects per dimension + summaries
    out_dir = Path("mixedlm_outputs"); out_dir.mkdir(exist_ok=True)
    for r in reports:
        (out_dir / f"fe_{r['dimension'].lower()}.csv").write_text(
            r["fe_params"].to_csv(index=True)
        )
        (out_dir / f"summary_{r['dimension'].lower()}.txt").write_text(r["summary"])

    logging.info("=== Done ===")

if __name__ == "__main__":
    main()

