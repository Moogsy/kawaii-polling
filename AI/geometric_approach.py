#!/usr/bin/env python3
# train_kawaii_geometry.py


import os
import sys

# 1) suppress glog INFO/WARNING & TF C++ logs
os.environ["GLOG_minloglevel"]    = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 2) redirect the OS-level stderr FD (fd 2) into a real log file
err_log = open("mediapipe_stderr.log", "w")
os.dup2(err_log.fileno(), 2)

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- Configuration Logging ---

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",handlers=[logging.StreamHandler(sys.stdout)]
    )

# --- Keypoint Extraction ---

def extract_keypoints(image_path: Path) -> np.ndarray | None:
    """
    Extrait les keypoints (x,y) normalisés via MediaPipe Pose.
    """
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        img = cv2.imread(str(image_path))
        if img is None:
            logging.warning(f"Cannot read image: {image_path}")
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
    if not results.pose_landmarks:
        logging.warning(f"No landmarks for: {image_path}")
        return None
    pts = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
    return np.array(pts)

# --- Geometric Feature Computation ---

def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def compute_geom_features(pts: np.ndarray) -> np.ndarray:
    idx = {
        'RSHO':12,'RELB':14,'RWRA':16,
        'LSHO':11,'LELB':13,'LWRA':15,
        'RHIP':24,'RKNE':26,'RANK':28,
        'LHIP':23,'LKNE':25,'LANK':27
    }
    feats = []
    # Joint angles
    feats.append(angle(pts[idx['RSHO']], pts[idx['RELB']], pts[idx['RWRA']]))
    feats.append(angle(pts[idx['LSHO']], pts[idx['LELB']], pts[idx['LWRA']]))
    feats.append(angle(pts[idx['RHIP']], pts[idx['RKNE']], pts[idx['RANK']]))
    feats.append(angle(pts[idx['LHIP']], pts[idx['LKNE']], pts[idx['LANK']]))
    # Limb/torso ratios
    def dist(u,v): return np.linalg.norm(u-v)
    ua_r = dist(pts[idx['RSHO']], pts[idx['RELB']])
    tor_r= dist(pts[idx['RSHO']], pts[idx['RHIP']]) + 1e-8
    ua_l = dist(pts[idx['LSHO']], pts[idx['LELB']])
    tor_l= dist(pts[idx['LSHO']], pts[idx['LHIP']]) + 1e-8
    feats.append(ua_r/tor_r)
    feats.append(ua_l/tor_l)
    # Symmetry
    feats.append(abs(feats[0]-feats[1]))
    feats.append(abs(feats[2]-feats[3]))
    # Center of gravity
    cog = pts.mean(axis=0)
    feats.extend([cog[0], cog[1]])
    return np.array(feats)

# --- Main Pipeline ---

def main():
    setup_logging()
    logging.info("=== Début pipeline géométrique ===")

    project_root = Path.cwd()
    agg_csv = project_root / "agg_phase2.csv"
    static_dir = project_root / "static"

    # Load aggregated scores
    df = pd.read_csv(agg_csv)
    logging.info(f"Loaded {len(df)} aggregated entries from {agg_csv.name}")

    # Ensure image_path column
    if 'image_path' not in df.columns:
        df['image_path'] = df.apply(
            lambda r: static_dir / r['Category'] / f"blurred_{r['ModelID']}.png",
            axis=1
        )
        logging.info("Reconstructed image_path column.")

    # Extract features
    X, y, cats = [], [], []
    for _, row in df.iterrows():
        img_path = Path(row['image_path'])
        if not img_path.exists():
            logging.warning(f"Missing: {img_path}")
            continue
        kpts = extract_keypoints(img_path)
        if kpts is None:
            continue
        feats = compute_geom_features(kpts)
        X.append(feats); y.append(row['kawaii_mean']); cats.append(row['Category'])

    X = np.vstack(X); y = np.array(y); cats = np.array(cats)
    logging.info(f"Computed features for {X.shape[0]} images")

    # Split
    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.2, stratify=cats, random_state=42
    )
    logging.info(f"Train/test sizes: {len(y_train)}/{len(y_test)}")

    # Train RF
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    logging.info("RF model trained")

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    logging.info(f"Results – MAE: {mae:.3f}, R2: {r2:.3f}")

if __name__ == '__main__':
    main()
