"""
Explores relationships between the three perceptual dimensions (Kawaii, Warmth,
Expressiveness).  compute_correlation_matrix() builds a Spearman or Pearson
correlation matrix at either the individual-rater level or the image-aggregate level.
plot_pairwise_correlation_scatter() draws a scatter plot for every pair of dimensions
with a LOWESS trend line and annotates each panel with Spearman ρ and p-value; jitter
is applied when working at the rater level to reduce overplotting on the Likert grid.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr
import statsmodels.api as sm

def compute_correlation_matrix(
    df: pd.DataFrame,
    level: str = "rater",
    method: str = "spearman"
) -> pd.DataFrame:
    if level == "rater":
        pivot = df.pivot_table(
            index=["Category", "Model", "RaterID"],
            columns="Rating",
            values="Score"
        )
    elif level == "image":
        pivot = df.pivot_table(
            index=["Category", "Model"],
            columns="Rating",
            values="Score",
            aggfunc="mean"
        )
    else:
        raise ValueError("`level` must be 'rater' or 'image'")
    return pivot.dropna().corr(method=method)

def plot_pairwise_correlation_scatter(df: pd.DataFrame, level: str):
    """
    Scatterplots avec courbes de tendance LOWESS pour toutes les paires de dimensions,
    avec coefficient de Spearman (ρ) et p-value.
    Utilise du jitter pour les données discrètes (niveau 'rater').
    """

    if level == "rater":
        pivot = df.pivot_table(
            index=["Category", "Model", "RaterID"],
            columns="Rating",
            values="Score"
        ).dropna().reset_index()
        title_suffix = "based on individual evaluations"
    elif level == "image":
        pivot = df.pivot_table(
            index=["Category", "Model"],
            columns="Rating",
            values="Score",
            aggfunc="mean"
        ).dropna().reset_index()
        title_suffix = "aggregated by pose-model pair"
    else:
        raise ValueError("`level` must be 'rater' or 'image'")

    rating_cols = ['Kawaii', 'Warmth', 'Expressiveness']
    pairs = list(combinations(rating_cols, 2))

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4), squeeze=False)
    jitter_strength = 0.1 if level == "rater" else 0.0

    for (i, (x_var, y_var)) in enumerate(pairs):
        ax = axes[0, i]
        x = pivot[x_var].values
        y = pivot[y_var].values

        # Jitter si applicable
        if jitter_strength > 0:
            x_j = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))
            y_j = y + np.random.uniform(-jitter_strength, jitter_strength, size=len(y))
        else:
            x_j, y_j = x, y

        # Corrélation
        rho, pval = spearmanr(x, y)

        # Scatterplot
        sns.scatterplot(x=x_j, y=y_j, ax=ax, color="orange", marker="x", edgecolor="w", s=40)

        # Courbe LOWESS
        smoothed = sm.nonparametric.lowess(y, x, frac=0.4)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=1.5, label="LOWESS")

        # Annotations
        ax.set_title(f"{x_var} vs {y_var}", fontsize=11)
        ax.text(0.05, 0.95, f"ρ = {rho:.2f}\np = {pval:.3f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9))

        # Axes
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.set_xticks(np.arange(1, 6))
        ax.set_yticks(np.arange(1, 6))
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Spearman correlations between perceptual dimensions\n({title_suffix})",
                 fontsize=15, fontweight='bold')

    fig.text(
        0.5, 0.01,
        "Note: Black line = LOWESS trend. Jitter added for Likert-scale clarity (rater level only).",
        ha='center', fontsize=9, style='italic', alpha=0.85
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

