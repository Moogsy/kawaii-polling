#!/usr/bin/env python3
import os

# suppress all glog INFO/WARNING messages (0=INFO,1=WARNING,2=ERROR,3=FATAL)
os.environ["GLOG_minloglevel"] = "2"
# suppress TensorFlow C++ logs (0=DEBUG,1=INFO,2=WARNING,3=ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Now disable Abseil (used by MediaPipe) warnings altogether
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# Prevent Abseil from printing its own init message
absl.logging._warn_preinit_stderr = False

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --- Configuration Logging ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# --- Feature Extraction ---
def extract_pixel_features(image_paths, size=(64,64)):
    logging.info(f"Extracting pixel features for {len(image_paths)} images")
    vectors = []
    for i, path in enumerate(image_paths, 1):
        try:
            img = Image.open(path).convert('L').resize(size)
            vectors.append(np.array(img).flatten())
        except Exception as e:
            logging.warning(f"Failed to process {path}: {e}")
        if i % 10 == 0:
            logging.info(f"Processed {i}/{len(image_paths)} images")
    X = np.stack(vectors)
    logging.info(f"Feature matrix: {X.shape}")
    return X

# --- Main Pipeline ---
def main():
    setup_logging()
    project_root = Path.cwd()
    agg_csv = project_root / 'agg_phase2.csv'
    static_dir = project_root / 'static'

    # Load aggregated scores
    df = pd.read_csv(agg_csv)
    logging.info(f"Loaded agg_phase2.csv with {len(df)} entries")

    # Ensure image_path column
    if 'image_path' not in df.columns:
        df['image_path'] = df.apply(
            lambda r: static_dir / r['Category'] / f"blurred_{r['ModelID']}.png",
            axis=1
        )
        logging.info("Reconstructed image_path column")

    # Filter existing images
    df['exists'] = df['image_path'].apply(lambda p: Path(p).exists())
    missing = len(df) - df['exists'].sum()
    logging.info(f"Images missing: {missing}; using {df['exists'].sum()} images")
    df = df[df['exists']].reset_index(drop=True)

    # Extract pixel features and target
    image_paths = df['image_path'].tolist()
    X = extract_pixel_features(image_paths)
    y = df['kawaii_mean'].values
    cats = df['Category'].values

    # PCA
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    logging.info(f"PCA retains {explained:.3f} variance with 50 components")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, stratify=cats, random_state=42
    )
    logging.info(f"Train/test split: {len(y_train)}/{len(y_test)} images")

    # Train RF baseline
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("RandomForest trained")

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Test MAE: {mae:.3f}, R2: {r2:.3f}")

if __name__ == '__main__':
    main()

