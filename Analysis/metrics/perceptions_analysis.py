import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

__all__ = (
    "compute_rater_bias_stats",
    "compute_interrater_consistency",
    "plot_rater_bias_stats",
    "plot_extreme_score_proportions",
    "plot_interrater_consistency"
)

def compute_rater_bias_stats(df: pd.DataFrame, rating: str = "Kawaii") -> pd.DataFrame:
    """
    Calcule pour chaque évaluateur :
      - mean_score  : moyenne des scores pour un critère donné
      - std_score   : écart-type
      - n           : nombre d'annotations
      - pct_low     : % de notes <=2 (sol)
      - pct_high    : % de notes >=4 (plafond)
    """
    sub = df[df["Rating"] == rating]
    group = sub.groupby("RaterID")["Score"]
    stats = group.agg(
        mean_score="mean",
        std_score="std",
        n="count"
    ).reset_index()  # type: ignore

    # Proportions extrêmes
    counts = sub.groupby(["RaterID", "Score"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(1,6), fill_value=0)  # scores 1–5 # type: ignore
    total = counts.sum(axis=1)
    stats["pct_low"]  = counts.loc[:, [1,2]].sum(axis=1) / total * 100  # type: ignore
    stats["pct_high"] = counts.loc[:, [4,5]].sum(axis=1) / total * 100  # type: ignore

    return stats

def compute_interrater_consistency(df: pd.DataFrame, rating: str = "Kawaii") -> pd.Series:
    """
    Calcule la consistance inter-évaluateurs via corrélations de Spearman.
    Retourne la série des corrélations moyennes pour chaque évaluateur.
    """
    sub = df[df["Rating"] == rating]
    pivot = sub.pivot_table(
        index=["Category", "Model"],
        columns="RaterID",
        values="Score"
    ).dropna(axis=1, how="any")  # type: ignore

    raters = pivot.columns
    corrs = pd.DataFrame(index=raters, columns=raters, dtype=float)

    for r1 in raters:
        for r2 in raters:
            corrs.at[r1, r2], _ = spearmanr(pivot[r1], pivot[r2])  # type: ignore

    # Moyenne des corrélations hors diagonale
    return corrs.apply(lambda row: row.drop(row.name).mean(), axis=1)  # type: ignore

def plot_extreme_score_proportions(
    df: pd.DataFrame,
    criteria: list[str] = ["Kawaii", "Warmth", "Expressiveness"]
) -> None:
    """
    Affiche, pour chaque critère, les proportions de scores extrêmes
    (Low ≤2 vs High ≥4) par évaluateur.
    """
    n = len(criteria)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]  # type: ignore

    for ax, crit in zip(axes, criteria):
        stats = compute_rater_bias_stats(df, rating=crit)
        stats = stats.sort_values("mean_score").reset_index(drop=True)
        raters = stats["RaterID"].astype(str)
        low    = stats["pct_low"].values
        high   = stats["pct_high"].values
        counts = stats["n"].values

        x = np.arange(len(raters))
        width = 0.4

        ax.bar(x - width/2, low,  width, label="Low (≤2)%",  color="lightcoral")
        ax.bar(x + width/2, high, width, label="High (≥4)%", color="seagreen")

        labels = [f"{r}\n(n={cnt})" for r, cnt in zip(raters, counts)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")  # type: ignore

        ax.set_title(f"Extreme Score Proportions – '{crit}'", fontsize=13, fontweight="semibold")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

    plt.show()

def plot_rater_bias_stats(
    df: pd.DataFrame,
    criteria: list[str] = ["Kawaii", "Warmth", "Expressiveness"]
) -> None:
    """
    Affiche, pour chaque critère :
      - Bars de mean_score triées par moyenne, colorées par quartile
      - Barres d’erreur (std_score) en arrière-plan (gris, fondu)
      - Lignes verticales Q1, Médiaine, Q3 avec légende
      - Sample size (n) dans les labels des rater
      - Marqueur ★ rouge pour les auto-évaluateurs (RaterID == Model)
      - Axe secondaire : % Low (≤2) et % High (≥4)
    """
    # Identify self-raters
    models = set(df["Model"].astype(str).unique())
    n = len(criteria)
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]  # type: ignore

    for ax, rating in zip(axes, criteria):
        stats = compute_rater_bias_stats(df, rating=rating)
        stats = stats.sort_values("mean_score").reset_index(drop=True)
        raters   = stats["RaterID"].astype(str).tolist()
        means    = stats["mean_score"].values
        stds     = stats["std_score"].values
        pct_low  = stats["pct_low"].values
        pct_high = stats["pct_high"].values
        counts   = stats["n"].values

        x = np.arange(len(raters))

        # Quartiles
        q1, q2, q3 = np.quantile(means, [0.25, 0.5, 0.75])
        idx_q1 = x[np.searchsorted(means, q1, 'right') - 1] + 0.5  # type: ignore
        idx_q2 = x[np.searchsorted(means, q2, 'right') - 1] + 0.5  # type: ignore
        idx_q3 = x[np.searchsorted(means, q3, 'right') - 1] + 0.5  # type: ignore

        # Colors by quartile
        boundaries = np.searchsorted(means, [q1, q2, q3])
        colors = [
            "lightblue" if i < boundaries[0] else
            "lightgreen" if i < boundaries[1] else
            "orange"     if i < boundaries[2] else
            "lightcoral"
            for i in range(len(means))
        ]

        # Plot bars and light errorbars
        ax.bar(x, means, color=colors, edgecolor='black')
        ax.errorbar(x, means, yerr=stds, fmt='none',
                    ecolor='gray', alpha=0.3, elinewidth=1, capsize=3)

        ax.set_title(f"Rater Bias – '{rating}'", fontsize=14, fontweight='bold')
        ax.set_ylabel("Mean Score")

        labels = [f"{r} (n={cnt})" for r, cnt in zip(raters, counts)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")  # type: ignore
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Quartile lines and legend
        l1 = ax.axvline(idx_q1, color='gray', linestyle='--', linewidth=1)
        l2 = ax.axvline(idx_q2, color='black', linestyle='-', linewidth=1)
        l3 = ax.axvline(idx_q3, color='gray', linestyle='--', linewidth=1)
        handles = [l1, l2, l3]
        labels_legend = [f"Q1={q1:.2f}", f"Median={q2:.2f}", f"Q3={q3:.2f}"]

        # Self-rater star
        star_h = None
        for i, r in enumerate(raters):
            if r in models:
                h, = ax.plot(x[i], means[i], marker='*', color='red', markersize=12)
                if star_h is None:
                    star_h = h
                    handles.append(h)
                    labels_legend.append("★ self-rater")

        ax.legend(handles, labels_legend, loc='upper left', fontsize=9, framealpha=0.8)

        # Secondary axis: extreme proportions
        ax2 = ax.twinx()
        ln1, = ax2.plot(x, pct_low,  '--o', color='blue',  label='% Low (≤2)')
        ln2, = ax2.plot(x, pct_high, '--^', color='red',   label='% High (≥4)')
        ax2.set_ylabel("Extreme Scores (%)", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(0, 100)
        ax2.legend(handles=[ln1, ln2], loc='upper right', fontsize=9)

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


