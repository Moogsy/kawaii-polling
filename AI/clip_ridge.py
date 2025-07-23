#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP + PCA + Ridge regression pipeline for predicting kawaii scores from images.

Requires: pip install git+https://github.com/openai/CLIP.git
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import clip  # OpenAI CLIP package
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_data(csv_path: Path, static_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns:
        df['image_path'] = (
            df['Category'].apply(lambda c: static_dir / c) /
            df['ModelID'].apply(lambda m: f"blurred_{m}.png")
        ).astype(str).apply(Path)
    return df


def embed_images(df: pd.DataFrame, model, preprocess, device: str) -> np.ndarray:
    embeddings = []
    model.to(device)
    model.eval()
    for img_path in df['image_path']:
        img = Image.open(img_path).convert('RGB')
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image_input)
        embeddings.append(feat.cpu().numpy().reshape(-1))
    return np.vstack(embeddings)


def plot_cv_scores(train_scores, test_scores):
    folds = np.arange(1, len(train_scores) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(folds, train_scores, 'o-', label='Train R²')
    plt.plot(folds, test_scores, 'o-', label='Test R²')
    plt.xlabel('Fold')
    plt.ylabel('R²')
    plt.title('Ridge: Train vs Test R² per GroupKFold')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    setup_logging()
    logging.info("=== CLIP + PCA + Ridge Kawaii Pipeline ===")

    project_root = Path.cwd()
    df = load_data(project_root / 'agg_phase2.csv', project_root / 'static')
    logging.info(f"Loaded {len(df)} samples")

    # Load CLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)

    # Compute image embeddings
    X = embed_images(df, clip_model, clip_preprocess, device)
    logging.info(f"Computed CLIP embeddings: {X.shape}")

    # PCA reduction
    pca = PCA(n_components=20, random_state=42)
    X = pca.fit_transform(X)
    logging.info(f"PCA reduced embeddings to: {X.shape}")

    # Raw target
    y = df['kawaii_mean'].values

    # GroupKFold on participants
    groups = pd.factorize(df['ModelID'].astype(str))[0]
    gkf = GroupKFold(n_splits=5)

    # Ridge regression CV
    ridge = Ridge(alpha=1.0)
    cv = cross_validate(
        ridge, X, y,
        cv=gkf.split(X, y, groups),
        scoring='r2',
        return_train_score=True,
        n_jobs=-1
    )
    train_r2 = cv['train_score']
    test_r2 = cv['test_score']
    logging.info(f"CV Train R²: {np.mean(train_r2):.3f} ± {np.std(train_r2):.3f}")
    logging.info(f"CV Test  R²: {np.mean(test_r2):.3f} ± {np.std(test_r2):.3f}")

    plot_cv_scores(train_r2, test_r2)

    # Final model fit and evaluation
    ridge.fit(X, y)
    y_pred = cross_val_predict(ridge, X, y, cv=gkf.split(X, y, groups), n_jobs=-1)
    overall_r2 = r2_score(y, y_pred)
    logging.info(f"Overall R² on raw target: {overall_r2:.3f}")

    # Coefficients plot
    coef_order = np.argsort(np.abs(ridge.coef_))[::-1]
    plt.figure(figsize=(6, 4))
    plt.barh([f"PC{idx+1}" for idx in coef_order], ridge.coef_[coef_order])
    plt.gca().invert_yaxis()
    plt.xlabel('Coefficient')
    plt.title('Ridge Coefficients on PCA components')
    plt.tight_layout()
    plt.show()

    # Save models
    torch.save(ridge, project_root / 'ridge_kawaii_model.pt')
    torch.save(pca, project_root / 'pca_clip.pt')
    torch.save(clip_model.state_dict(), project_root / 'clip_vit_base_state_dict.pt')

    logging.info("Pipeline complete. Models and PCA saved.")


if __name__ == '__main__':
    main()

