#!/usr/bin/env python3
# train_kawaii_geometry.py

import os
import sys

# 1) suppress glog INFO/WARNING & TF C++ logs
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 2) redirect the OS-level stderr FD (fd 2) into a real log file
err_log = open("openpose_stderr.log", "w")
os.dup2(err_log.fileno(), 2)

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

# OpenPose imports
from openpose import pyopenpose as op

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

# --- OpenPose Initialization ---

def init_openpose():
    params = dict()
    params["model_folder"] = "models/"
    params["number_people_max"] = 1
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()
    return op_wrapper

# --- Pose Normalization Functions (unchanged) ---

def angle_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
    vec = p1 - p2
    return np.degrees(np.arctan2(vec[1], vec[0]))

def rotate_points(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    rot_matrix = np.array([[c, -s], [s, c]])
    return pts @ rot_matrix.T

def pca_orientation(pts: np.ndarray) -> float:
    pts_centered = pts - pts.mean(axis=0)
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
    if angle < 0:
        angle += 180
    return angle

def normalize_pose(pts: np.ndarray, conf: np.ndarray) -> np.ndarray:
    idx = {
        'RSHO': 2, 'LSHO': 5,
        'RHIP': 9, 'LHIP': 12,
    }

    # mid-hip and mid-shoulder
    mid_hip = (pts[idx['RHIP']] + pts[idx['LHIP']]) / 2
    mid_shoulder = (pts[idx['RSHO']] + pts[idx['LSHO']]) / 2

    # translate
    pts_centered = pts - mid_hip
    # scale
    torso_height = np.linalg.norm(mid_shoulder - mid_hip)
    if torso_height < 1e-6:
        torso_height = 1.0
    pts_scaled = pts_centered / torso_height

    # weighted rotation from shoulders and hips
    def line_angle_conf(p1,p2,c1,c2):
        if c1>0 and c2>0:
            ang = angle_between_points(p1,p2)
            weight = (c1+c2)/2
            return ang, weight
        return None, 0.0

    shoulder_ang, shoulder_w = line_angle_conf(
        pts[idx['LSHO']], pts[idx['RSHO']], conf[idx['LSHO']], conf[idx['RSHO']]
    )
    hip_ang, hip_w = line_angle_conf(
        pts[idx['LHIP']], pts[idx['RHIP']], conf[idx['LHIP']], conf[idx['RHIP']]
    )
    if shoulder_ang is None: shoulder_ang, shoulder_w = 0.0, 0.0
    if hip_ang is None: hip_ang, hip_w = 0.0, 0.0
    total_w = shoulder_w + hip_w
    if total_w > 0:
        target_ang = (shoulder_ang*shoulder_w + hip_ang*hip_w)/total_w
    else:
        target_ang = pca_orientation(pts_scaled)

    norm_pts = rotate_points(pts_scaled, -target_ang)
    # ensure upright
    mid_hip_r = (norm_pts[idx['RHIP']] + norm_pts[idx['LHIP']]) / 2
    mid_sh_r = (norm_pts[idx['RSHO']] + norm_pts[idx['LSHO']]) / 2
    if (mid_sh_r - mid_hip_r)[1] < 0:
        norm_pts[:,1] *= -1
    return norm_pts

# --- Keypoint Extraction via OpenPose ---

def extract_keypoints(image_path: Path, op_wrapper) -> tuple[np.ndarray, np.ndarray]|
 None:
    img = cv2.imread(str(image_path))
    if img is None:
        logging.warning(f"Cannot read image: {image_path}")
        return None, None
    datum = op.Datum()
    datum.cvInputData = img
    op_wrapper.emplaceAndPop([datum])
    keypoints = datum.poseKeypoints
    if keypoints is None or len(keypoints.shape) < 3:
        logging.warning(f"No pose detected: {image_path}")
        return None, None
    pts_conf = keypoints[0]  # first person: shape (25,3)
    pts = pts_conf[:, :2]
    conf = pts_conf[:, 2]
    # normalize coordinates to [0,1] based on image size
    h, w = img.shape[:2]
    pts[:, 0] /= w
    pts[:, 1] /= h
    return pts, conf

# --- Geometric Feature Computation (unchanged) ---

def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))

def compute_geom_features(pts: np.ndarray) -> np.ndarray:
    idx = {
        'RSHO':2,'RELB':3,'RWRA':4,
        'LSHO':5,'LELB':6,'LWRA':7,
        'RHIP':9,'RKNE':10,'RANK':11,
        'LHIP':12,'LKNE':13,'LANK':14
    }
    feats = []
    feats.append(angle(pts[idx['RSHO']], pts[idx['RELB']], pts[idx['RWRA']]))
    feats.append(angle(pts[idx['LSHO']], pts[idx['LELB']], pts[idx['LWRA']]))
    feats.append(angle(pts[idx['RHIP']], pts[idx['RKNE']], pts[idx['RANK']]))
    feats.append(angle(pts[idx['LHIP']], pts[idx['LKNE']], pts[idx['LANK']]))
    def dist(u,v): return np.linalg.norm(u-v)
    ua_r = dist(pts[idx['RSHO']], pts[idx['RELB']])
    tor_r = dist(pts[idx['RSHO']], pts[idx['RHIP']]) + 1e-8
    ua_l = dist(pts[idx['LSHO']], pts[idx['LELB']])
    tor_l = dist(pts[idx['LSHO']], pts[idx['LHIP']]) + 1e-8
    feats.append(ua_r/tor_r)
    feats.append(ua_l/tor_l)
    feats.append(abs(feats[0]-feats[1]))
    feats.append(abs(feats[2]-feats[3]))
    cog = pts.mean(axis=0)
    feats.extend([cog[0], cog[1]])
    return np.array(feats)

# --- Main Pipeline ---

def main():
    setup_logging()
    logging.info("=== Début pipeline géométrique avec OpenPose ===")

    op_wrapper = init_openpose()
    project_root = Path.cwd()
    agg_csv = project_root / "agg_phase2.csv"
    static_dir = project_root / "static"

    df = pd.read_csv(agg_csv)
    logging.info(f"Loaded {len(df)} entries from {agg_csv.name}")

    if 'image_path' not in df.columns:
        df['image_path'] = df.apply(
            lambda r: static_dir / r['Category'] / f"blurred_{r['ModelID']}.png",
            axis=1
        )
        logging.info("Reconstructed image_path column.")

    X, y, cats = [], [], []
    for _, row in df.iterrows():
        img_path = Path(row['image_path'])
        if not img_path.exists():
            logging.warning(f"Missing: {img_path}")
            continue
        kpts, conf = extract_keypoints(img_path, op_wrapper)
        if kpts is None or conf is None:
            continue
        kpts_norm = normalize_pose(kpts, conf)
        feats = compute_geom_features(kpts_norm)
        X.append(feats); y.append(row['kawaii_mean']); cats.append(row['Category'])

    X = np.vstack(X); y = np.array(y); cats = np.array(cats)
    logging.info(f"Computed features for {X.shape[0]} images")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=cats, random_state=42
    )
    logging.info(f"Train/test sizes: {len(y_train)}/{len(y_test)}")

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    logging.info("RF model trained")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Results – MAE: {mae:.3f}, R2: {r2:.3f}")

if __name__ == '__main__':
    main()

