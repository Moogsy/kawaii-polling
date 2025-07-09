import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

__all__ = (
    "compute_rater_bias_stats",
    "compute_interrater_consistency",
    "plot_rater_bias_stats",
    "plot_interrater_consistency"
)

def compute_rater_bias_stats(df: pd.DataFrame, rating: str = "Kawaii") -> pd.DataFrame:
    """
    Calcule pour chaque évaluateur :
      - mean_score      : moyenne des scores pour un critère donné
      - std_score       : écart-type
      - n               : nombre d'annotations
      - pct_low         : % de notes <=2 (sol)
      - pct_high        : % de notes >=4 (plafond)
    """
    sub = df[df["Rating"] == rating]
    group = sub.groupby("RaterID")["Score"]
    stats = group.agg(
        mean_score="mean",
        std_score="std",
        n="count"
    ).reset_index()
    
    # Proportions extrêmes
    counts = sub.groupby(["RaterID", "Score"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(6), fill_value=0)  # scores 0–5
    total = counts.sum(axis=1)
    stats["pct_low"] = counts.loc[:, [0,1,2]].sum(axis=1) / total * 100
    stats["pct_high"] = counts.loc[:, [4,5]].sum(axis=1) / total * 100
    
    return stats

def compute_interrater_consistency(df: pd.DataFrame, rating: str = "Kawaii") -> pd.Series:
    """
    Calcule la consistance inter-évaluateurs via corrélations de Spearman.
    Retourne la série des corrélations moyennes pour chaque évaluateur
    (moyenne des corrélations avec tous les autres).
    """
    sub = df[df["Rating"] == rating]
    pivot = sub.pivot_table(
        index=["Category", "Model"],
        columns="RaterID",
        values="Score"
    ).dropna(axis=1, how="any")
    
    raters = pivot.columns
    corrs = pd.DataFrame(index=raters, columns=raters, dtype=float)
    
    for i, r1 in enumerate(raters):
        for r2 in raters:
            corrs.at[r1, r2], _ = spearmanr(pivot[r1], pivot[r2])
    
    # moyenne des corrélations hors diagonale
    return corrs.apply(lambda row: row.drop(row.name).mean(), axis=1)


def plot_rater_bias_stats(
    df,
    criteria=["Kawaii", "Warmth", "Expressiveness"]
):
    """
    Enhanced rater bias plots for multiple criteria:
      - One subplot per criterion
      - Bars of mean_score (sorted ascending) colored by quartile
      - Light, grey errorbars (std_score) in background
      - Vertical lines marking Q1, median, Q3 (legend with values)
      - Sample size shown next to each rater name on the x-axis
      - Secondary axis: line plot of % high scores (≥4)
    """
    n = len(criteria)
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, rating in zip(axes, criteria):
        # Compute and sort per-rater stats
        stats = compute_rater_bias_stats(df, rating=rating)
        stats = stats.sort_values("mean_score").reset_index(drop=True)
        raters = stats["RaterID"].astype(str)
        means = stats["mean_score"].values
        stds = stats["std_score"].values
        pct_high = stats["pct_high"].values
        counts = stats["n"].values

        # Quartile values & positions
        q1, q2, q3 = np.quantile(means, [0.25, 0.5, 0.75])
        idx_q1 = np.searchsorted(means, q1, 'right') - 0.5
        idx_q2 = np.searchsorted(means, q2, 'right') - 0.5
        idx_q3 = np.searchsorted(means, q3, 'right') - 0.5

        # Colors by quartile
        colors = [
            "lightblue" if i < idx_q1+0.5 else
            "lightgreen" if i < idx_q2+0.5 else
            "orange"     if i < idx_q3+0.5 else
            "lightcoral"
            for i in range(len(means))
        ]

        # Bars + light errorbars
        ax.bar(raters, means, color=colors, edgecolor='black')
        ax.errorbar(raters, means, yerr=stds, fmt='none',
                    ecolor='gray', alpha=0.3, elinewidth=1, capsize=3)

        ax.set_title(f"Rater Bias – '{rating}'", fontsize=14, fontweight='bold')
        ax.set_ylabel("Mean Score")

        # Sample size in tick labels
        labels = [f"{r} (n={cnt})" for r, cnt in zip(raters, counts)]
        ax.set_xticks(raters)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Quartile lines + legend
        l1 = ax.axvline(idx_q1, color='gray', linestyle='--', linewidth=1)
        l2 = ax.axvline(idx_q2, color='black', linestyle='-', linewidth=1)
        l3 = ax.axvline(idx_q3, color='gray', linestyle='--', linewidth=1)
        ax.legend([l1, l2, l3],
                  [f"Q1 = {q1:.2f}", f"Median = {q2:.2f}", f"Q3 = {q3:.2f}"],
                  loc='upper left', fontsize=9, framealpha=0.8)

        # Secondary axis: % high scores
        ax2 = ax.twinx()
        ax2.plot(raters, pct_high, '-o', color='red', label='% High (≥4)')
        ax2.set_ylabel("% High Scores (≥4)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right', fontsize=9)

    plt.show()


def plot_interrater_consistency(consistency: pd.Series) -> None:
    """
    Affiche la distribution des corrélations moyennes de Spearman
    entre évaluateurs (consistency par rater).
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(consistency, bins=10, edgecolor="black")
    ax.set_title("Inter-rater consistency (mean Spearman ρ)")
    ax.set_xlabel("Mean Spearman ρ")
    ax.set_ylabel("Number of Raters")
    plt.show()

