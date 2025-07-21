#!/usr/bin/env python3
# train_kawaii_geometry.py

import os
import sys

# 1) suppress glog INFO/WARNING & TF C++ logs
os.environ["GLOG_minloglevel"] = "2"
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
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# --- Pose Normalization Functions ---

def angle_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute angle (in degrees) of vector p2->p1 relative to horizontal axis."""
    vec = p1 - p2
    return np.degrees(np.arctan2(vec[1], vec[0]))

def rotate_points(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate 2D points by angle_deg around origin."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    rot_matrix = np.array([[c, -s], [s, c]])
    return pts @ rot_matrix.T

def pca_orientation(pts: np.ndarray) -> float:
    """Estimate main orientation angle (degrees) of pts using PCA."""
    pts_centered = pts - pts.mean(axis=0)
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
    # Make sure angle points upwards (y > 0)
    if angle < 0:
        angle += 180
    return angle

def normalize_pose(pts: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """
    Normalize pose keypoints:
    - Translate so mid-hip at origin
    - Scale by torso height (mid-shoulder to mid-hip)
    - Rotate to minimize weighted angle from shoulders and hips lines
    - Fallback to PCA orientation if needed

    Parameters:
    - pts: np.ndarray shape (33, 2), x,y normalized keypoints from MediaPipe
    - visibility: np.ndarray shape (33,), visibility scores [0..1] for each keypoint

    Returns:
    - normalized_pts: np.ndarray shape (33, 2)
    """

    idx = {
        'RSHO': 12, 'LSHO': 11,
        'RHIP': 24, 'LHIP': 23,
        'RELB': 14, 'LELB': 13,
        'RWRA': 16, 'LWRA': 15,
        'NOSE': 0,
    }

    # Compute mid-hip and mid-shoulder points
    mid_hip = (pts[idx['RHIP']] + pts[idx['LHIP']]) / 2
    mid_shoulder = (pts[idx['RSHO']] + pts[idx['LSHO']]) / 2

    # Translate pts to mid-hip origin
    pts_centered = pts - mid_hip

    # Compute torso height (distance mid-shoulder to mid-hip)
    torso_vec = mid_shoulder - mid_hip
    torso_height = np.linalg.norm(torso_vec)
    if torso_height < 1e-6:
        torso_height = 1.0  # Avoid division by zero, fallback to 1

    pts_scaled = pts_centered / torso_height

    # Compute shoulder and hip line angles and visibilities
    def line_angle_and_vis(p1, p2, v1, v2):
        if v1 > 0 and v2 > 0:
            ang = angle_between_points(p1, p2)
            vis = (v1 + v2) / 2
            return ang, vis
        else:
            return None, 0.0

    shoulder_angle, shoulder_vis = line_angle_and_vis(
        pts[idx['LSHO']], pts[idx['RSHO']], visibility[idx['LSHO']], visibility[idx['RSHO']]
    )
    hip_angle, hip_vis = line_angle_and_vis(
        pts[idx['LHIP']], pts[idx['RHIP']], visibility[idx['LHIP']], visibility[idx['RHIP']]
    )

    # If no visibility, fallback to zero angles and zero visibility
    if shoulder_angle is None:
        shoulder_angle, shoulder_vis = 0.0, 0.0
    if hip_angle is None:
        hip_angle, hip_vis = 0.0, 0.0

    total_vis = shoulder_vis + hip_vis

    if total_vis > 0:
        # Weighted average angle minimizing squared error
        target_angle = (shoulder_angle * shoulder_vis + hip_angle * hip_vis) / total_vis
    else:
        # Fallback PCA orientation on scaled points
        target_angle = pca_orientation(pts_scaled)

    # Rotate points by negative target_angle to align horizontally
    normalized_pts = rotate_points(pts_scaled, -target_angle)

    # Optional: Flip vertically if torso axis points downward after rotation
    mid_hip_rot = (normalized_pts[idx['RHIP']] + normalized_pts[idx['LHIP']]) / 2
    mid_shoulder_rot = (normalized_pts[idx['RSHO']] + normalized_pts[idx['LSHO']]) / 2
    torso_vec_rot = mid_shoulder_rot - mid_hip_rot
    if torso_vec_rot[1] < 0:
        normalized_pts[:, 1] *= -1

    return normalized_pts

# --- Keypoint Extraction ---

def extract_keypoints(image_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Extrait les keypoints (x,y) normalisés via MediaPipe Pose et leur visibilité.
    """
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        img = cv2.imread(str(image_path))
        if img is None:
            logging.warning(f"Cannot read image: {image_path}")
            return None, None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
    if not results.pose_landmarks:
        logging.warning(f"No landmarks for: {image_path}")
        return None, None
    pts = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])
    visibility = np.array([lm.visibility for lm in results.pose_landmarks.landmark])
    return pts, visibility

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
        kpts, visibility = extract_keypoints(img_path)
        if kpts is None or visibility is None:
            continue

        # Normalize pose before feature extraction
        kpts_norm = normalize_pose(kpts, visibility)

        feats = compute_geom_features(kpts_norm)
        X.append(feats); y.append(row['kawaii_mean']); cats.append(row['Category'])

    X = np.vstack(X); y = np.array(y); cats = np.array(cats)
    logging.info(f"Computed features for {X.shape[0]} images")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
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
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Results – MAE: {mae:.3f}, R2: {r2:.3f}")

if __name__ == '__main__':
    main()

