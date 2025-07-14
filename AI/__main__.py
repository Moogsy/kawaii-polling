#!/usr/bin/env python3
# train_kawaii_geometry.py

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
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# --- Keypoint Extraction ---
def extract_keypoints(image_path: Path) -> np.ndarray | None:
    """
    Extrait les keypoints (x,y) normalisés via MediaPipe Pose.
    Retourne un array (n_points, 2) ou None si échec.
    """
    pose = mp.solutions.pose.Pose(static_image_mode=True)
    img = cv2.imread(str(image_path))
    if img is None:
        logging.warning(f"Cannot read image: {image_path}")
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    pose.close()
    if not results.pose_landmarks:
        logging.warning(f"No landmarks detected for: {image_path}")
        return None
    pts = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
    return np.array(pts)

# --- Geometric Feature Computation ---
def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calcule l’angle (en degrés) au point b formé par a-b-c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def compute_geom_features(pts: np.ndarray) -> np.ndarray:
    """
    À partir d’un array (n_points,2), calcule un vecteur de features géométriques :
    - Angles (coudes, genoux)
    - Ratios de longueurs (bras/taille)
    - Symétrie gauche/droite
    - Centre de gravité
    """
    # Indices keypoints MediaPipe
    idx = {
        'RSHO': 12, 'RELB': 14, 'RWRA': 16,
        'LSHO': 11, 'LELB': 13, 'LWRA': 15,
        'RHIP': 24, 'RKNE': 26, 'RANK': 28,
        'LHIP': 23, 'LKNE': 25, 'LANK': 27
    }
    feats = []
    # Angles extrémités
    feats.append(angle(pts[idx['RSHO']], pts[idx['RELB']], pts[idx['RWRA']]))  # coude droit
    feats.append(angle(pts[idx['LSHO']], pts[idx['LELB']], pts[idx['LWRA']]))  # coude gauche
    feats.append(angle(pts[idx['RHIP']], pts[idx['RKNE']], pts[idx['RANK']]))  # genou droit
    feats.append(angle(pts[idx['LHIP']], pts[idx['LKNE']], pts[idx['LANK']]))  # genou gauche
    # Ratios de longueur bras/taille
    def dist(u,v): return np.linalg.norm(u-v)
    upper_arm_r = dist(pts[idx['RSHO']], pts[idx['RELB']])
    torso_r     = dist(pts[idx['RSHO']], pts[idx['RHIP']]) + 1e-8
    upper_arm_l = dist(pts[idx['LSHO']], pts[idx['LELB']])
    torso_l     = dist(pts[idx['LSHO']], pts[idx['LHIP']]) + 1e-8
    feats.append(upper_arm_r / torso_r)
    feats.append(upper_arm_l / torso_l)
    # Symétrie angles
    feats.append(abs(feats[0] - feats[1]))  # diff coude
    feats.append(abs(feats[2] - feats[3]))  # diff genou
    # Centre de gravité
    cog = pts.mean(axis=0)
    feats.append(cog[0])  # x
    feats.append(cog[1])  # y
    return np.array(feats)

# --- Main Pipeline ---
def main():
    setup_logging()
    logging.info("=== Début du pipeline géométrique ===")

    # Chargement des données agrégées
    df = pd.read_csv('agg_phase2.csv')
    logging.info(f"Loaded aggregated CSV: {len(df)} records")

    # Extraction et calcul des features
    X, y, cats = [], [], []
    for _, row in df.iterrows():
        img_path = Path(row['image_path'])
        if not img_path.exists():
            logging.warning(f"Missing image: {img_path}")
            continue
        kpts = extract_keypoints(img_path)
        if kpts is None:
            continue
        feats = compute_geom_features(kpts)
        X.append(feats)
        y.append(row['kawaii_mean'])
        cats.append(row['Category'])
    X = np.vstack(X)
    y = np.array(y)
    cats = np.array(cats)
    logging.info(f"Computed geometric features for {X.shape[0]} images")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=cats, random_state=42
    )
    logging.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Entraînement RandomForest
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    logging.info("RandomForest trained")

    # Évaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    logging.info(f"Keypoints RF – Test MAE: {mae:.3f}, R²: {r2:.3f}")

if __name__ == "__main__":
    main()

