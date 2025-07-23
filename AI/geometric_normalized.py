#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a RandomForest on pose geometry features to predict kawaii scores.
"""
import logging
import os
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate, cross_val_predict
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Global feature names must match output of compute_geom_features
FEATURE_NAMES = [
    'Elbow_Min_Angle', 'Elbow_Delta_Angle',
    'Knee_Min_Angle', 'Knee_Delta_Angle',
    'Arm_Mean_Ratio',
    'COG_X', 'COG_Y',
    'Head_Tilt', 'Head_Rot',
]
# -----------------------------------------------------------------------------

# Suppress MediaPipe/TensorFlow logs and redirect stderr
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
err_log = open("mediapipe_stderr.log", "w")
os.dup2(err_log.fileno(), 2)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class PoseExtractor(ABC):
    """
    Interface for extracting 2D keypoints and visibility scores from an image.
    """

    @abstractmethod
    def extract(self, image_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pass


class MediaPipePoseExtractor(PoseExtractor):
    """
    Extracts pose landmarks using MediaPipe's static image mode.
    """

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)

    def extract(self, image_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        img = cv2.imread(str(image_path))
        if img is None:
            logging.warning(f"Cannot read image: {image_path}")
            return None, None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            logging.warning(f"No landmarks for: {image_path}")
            return None, None

        pts = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])
        vis = np.array([lm.visibility for lm in results.pose_landmarks.landmark])
        return pts, vis


# Utility functions for geometry

def angle_between(p1: np.ndarray, p2: np.ndarray) -> float:
    vec = p1 - p2
    return np.degrees(np.arctan2(vec[1], vec[0]))


def rotate(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T


def pca_orientation(pts: np.ndarray) -> float:
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(axis[1], axis[0]))
    return angle + 180 if angle < 0 else angle


def normalize_pose(pts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    idx = {'LSHO': 11, 'RSHO': 12, 'LHIP': 23, 'RHIP': 24}
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

    total_w = sw + hw
    target = (sa * sw + ha * hw) / total_w if total_w > 0 else pca_orientation(scaled)
    norm = rotate(scaled, -target)

    hip_center = (norm[idx['LHIP']] + norm[idx['RHIP']]) / 2
    sho_center = (norm[idx['LSHO']] + norm[idx['RSHO']]) / 2
    if (sho_center - hip_center)[1] < 0:
        norm[:, 1] *= -1

    return norm


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    cos_v = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0)))


def compute_geom_features(pts: np.ndarray) -> np.ndarray:
    idx = {
        'RSHO': 12, 'RELB': 14, 'RWRA': 16,
        'LSHO': 11, 'LELB': 13, 'LWRA': 15,
        'RHIP': 24, 'RKNE': 26, 'RANK': 28,
        'LHIP': 23, 'LKNE': 25, 'LANK': 27,
        'LEAR': 7,  'REAR': 8,  'NOSE': 0
    }
    dist = lambda u, v: np.linalg.norm(u - v)

    # Joint angles
    raw_elbow = [
        angle(pts[idx['RSHO']], pts[idx['RELB']], pts[idx['RWRA']]),
        angle(pts[idx['LSHO']], pts[idx['LELB']], pts[idx['LWRA']])
    ]
    raw_knee = [
        angle(pts[idx['RHIP']], pts[idx['RKNE']], pts[idx['RANK']]),
        angle(pts[idx['LHIP']], pts[idx['LKNE']], pts[idx['LANK']])
    ]
    elbow_min = min(raw_elbow)
    elbow_delta = max(raw_elbow) - elbow_min
    knee_min = min(raw_knee)
    knee_delta = max(raw_knee) - knee_min

    # Morphology: mean arm-to-torso ratio (upper arm)
    r_ratio = dist(pts[idx['RSHO']], pts[idx['RELB']])
    l_ratio = dist(pts[idx['LSHO']], pts[idx['LELB']])
    arm_mean_ratio = (r_ratio + l_ratio) / 2

    # Center of gravity
    cog_x, cog_y = pts.mean(axis=0)

    # Head features
    head_tilt = angle_between(pts[idx['LEAR']], pts[idx['REAR']])
    head_rot = dist(pts[idx['NOSE']], pts[idx['LSHO']]) / (dist(pts[idx['NOSE']], pts[idx['RSHO']]) + 1e-8)

    feats = [
        elbow_min, elbow_delta,
        knee_min, knee_delta,
        arm_mean_ratio,
        cog_x, cog_y,
        head_tilt, head_rot
    ]
    arr = np.array(feats)
    assert arr.shape[0] == len(FEATURE_NAMES), (
        f"Expected {len(FEATURE_NAMES)} features, got {arr.shape[0]}"
    )
    return arr


def _plot_feature_importance(importances: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    order = np.argsort(importances)[::-1]
    plt.barh([FEATURE_NAMES[i] for i in order], importances[order])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()


def _plot_regression(y: np.ndarray, y_pred: np.ndarray, overall_r2: float) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y, y_pred, alpha=0.7, edgecolors='k')
    mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], '--', linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Pred vs Actual (R2={overall_r2:.2f})")
    plt.tight_layout()
    plt.show()


def main():
    setup_logging()
    logging.info("=== Starting geometric kawaii pipeline ===")

    extractor = MediaPipePoseExtractor()
    logging.info(f"Using extractor: {extractor.__class__.__name__}")

    project_root = Path.cwd()
    df = pd.read_csv(project_root / "agg_phase2.csv")
    logging.info(f"Loaded {len(df)} rows from agg_phase2.csv")

    if 'image_path' not in df:
        df['image_path'] = (
            project_root / 'static' /
            df['Category'] /
            df['ModelID'].apply(lambda m: f"blurred_{m}.png")
        )

    # Compute residuals to remove person and pose biases
    df['person_mean'] = df.groupby('ModelID')['kawaii_mean'].transform('mean')
    df['pose_mean'] = df.groupby('Category')['kawaii_mean'].transform('mean')
    overall_mean = df['kawaii_mean'].mean()
    df['residual'] = (
        df['kawaii_mean']
        - df['person_mean']
        - df['pose_mean']
        + overall_mean
    )
    X, y, model_ids = [], [], []
    for _, row in df.iterrows():
        path = Path(row['image_path'])
        if not path.exists():
            logging.warning(f"Missing file: {path}")
            continue

        pts, vis = extractor.extract(path)
        if pts is None:
            continue

        norm = normalize_pose(pts, vis)
        X.append(compute_geom_features(norm))
        y.append(row['residual'])
        model_ids.append(row['ModelID'])

    X = np.vstack(X)
    y = np.array(y)
    logging.info(f"Computed features for {X.shape[0]} images")

    assert X.shape[1] == len(FEATURE_NAMES), (
        f"Feature dimension {X.shape[1]} does not match FEATURE_NAMES {len(FEATURE_NAMES)}"
    )

    from sklearn.model_selection import GroupKFold
    groups = pd.factorize(model_ids)[0]
    gkf = GroupKFold(n_splits=len(np.unique(groups)))

    model = RandomForestRegressor(
            n_estimators=200, 
            random_state=42
    )
    scoring = {'r2': 'r2', 'neg_mse': 'neg_mean_squared_error'}

    cv_results = cross_validate(
        model, X, y,
        cv=gkf.split(X, y, groups),
        scoring=scoring,
        return_train_score=True
    )
    train_r2 = cv_results['train_r2']
    test_r2 = cv_results['test_r2']
    folds = np.arange(1, len(train_r2) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(folds, train_r2, marker='o', label='Train R2')
    plt.plot(folds, test_r2, marker='o', label='Test R2')
    plt.xlabel('Fold')
    plt.ylabel('R²')
    plt.title('Train vs Test R² per GroupKFold')
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.fit(X, y)
    y_pred = cross_val_predict(model, X, y, cv=gkf.split(X, y, groups))
    overall_r2 = r2_score(y, y_pred)

    importances = model.feature_importances_
    _plot_feature_importance(importances)
    _plot_regression(y, y_pred, overall_r2)


if __name__ == '__main__':
    main()

