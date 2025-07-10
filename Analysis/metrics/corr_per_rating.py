import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    return pivot.dropna().corr(method=method) # type: ignore


def plot_dual_correlation_matrices(
    df: pd.DataFrame,
    method: str = "spearman",
    vmin: float = 0.5,
    vmax: float = 1.0,
    cmap: str = "coolwarm"
) -> None:
    """
    Affiche deux heatmaps côte à côte des corrélations Spearman entre types
    de notation :
      - Gauche : niveau d'évaluation individuel ('rater')
      - Droite : niveau agrégé par image ('image')
    """
    corr_rater = compute_correlation_matrix(df, level="rater", method=method)
    corr_image = compute_correlation_matrix(df, level="image", method=method)

    labels = corr_rater.columns.tolist()
    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    for ax, corr, level in zip(
        axes,
        [corr_rater, corr_image],
        ["Rater-level", "Image-level"]
    ):
        im = ax.imshow(corr.values, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr.iat[i, j]
                text_color = "white" if abs(val) > (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", color=text_color)
        ax.set_title(f"Spearman correlation between rating types\n({level})",
                     fontsize=12, pad=10)

    # Barre de couleur partagée
    fig.colorbar(im, ax=axes, orientation="vertical",  # type: ignore
        fraction=0.05, pad=0.02, label="Spearman ρ")
    plt.show()

