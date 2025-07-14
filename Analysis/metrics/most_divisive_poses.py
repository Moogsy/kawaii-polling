import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

__all__ = ("compute_divisiveness_per_rating", "compute_image_path", "plot_divisiveness_per_rating_percentage")

def compute_divisiveness_per_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque critère de notation ('Rating'), identifie l'image
    (Category, Model) la plus et la moins clivante d’après écart‑type.
    """
    stds = (
        df
        .groupby(["Rating", "Category", "Model"])["Score"]
        .std(ddof=0)
        .reset_index(name="std")
    )
    idx_max = stds.groupby("Rating")["std"].idxmax()
    idx_min = stds.groupby("Rating")["std"].idxmin()

    most = stds.loc[idx_max].rename(columns={
        "Category": "most_divisive_category",
        "Model": "most_divisive_model",
        "std": "most_std"
    })
    least = stds.loc[idx_min].rename(columns={
        "Category": "least_divisive_category",
        "Model": "least_divisive_model",
        "std": "least_std"
    })

    return pd.merge(
        most[["Rating", "most_divisive_category", "most_divisive_model", "most_std"]],
        least[["Rating", "least_divisive_category", "least_divisive_model", "least_std"]],
        on="Rating"
    )

def compute_image_path(entry: tuple[str, str], images_folder: Path) -> str:
    """
    Génère le chemin vers l'image à partir de (Category, Model),
    suivant le pattern : ../static/<category>/blurred_<model>.png
    """
    category, model = entry
    safe_model = "".join(c if c.isalnum() else "_" for c in model)
    return os.path.join(images_folder, category, f"blurred_{safe_model}.png")

def plot_divisiveness_per_rating_percentage(
    df: pd.DataFrame,
    *,
    images_folder: Path
) -> None:
    """
    Pour chaque critère :
      - Image la plus clivante + bar chart en % des scores (0–5)
      - Image la moins clivante + bar chart en % des scores (0–5)
    Affiche également N total dans le titre du bar chart.
    """
    div = compute_divisiveness_per_rating(df)
    ratings = div["Rating"].tolist()
    n = len(ratings)
    score_levels = list(range(1, 6))  # 0,1,2,3,4,5

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), constrained_layout=True)

    for i, row in div.iterrows():
        rating = row["Rating"]

        for col_idx, (prefix, label) in enumerate([
            ("most_divisive", "Most divisive"),
            ("least_divisive", "Least divisive")
        ]):
            cat = row[f"{prefix}_category"]
            mod = row[f"{prefix}_model"]
            std_val = row[f"{prefix.replace('divisive', 'std')}"]
            entry = (cat, mod)

            # 1) Affiche l'image
            ax_img = axes[i, col_idx * 2]
            img = plt.imread(compute_image_path(entry, images_folder))
            ax_img.imshow(img); ax_img.axis("off")
            ax_img.set_title(f"{rating}\n{label}\nσ={std_val:.2f}")

            # 2) Bar chart des scores en % pour 0–5
            scores = df.loc[
                (df["Rating"]==rating) &
                (df["Category"]==cat) &
                (df["Model"]==mod),
                "Score"
            ]
            counts = scores.value_counts().reindex(score_levels, fill_value=0)
            total = counts.sum()
            percentages = counts / total * 100

            ax_bar = axes[i, col_idx * 2 + 1]
            ax_bar.bar([str(s) for s in score_levels], percentages.values, edgecolor='black')
            ax_bar.set_ylim(0, 100)
            ax_bar.set_xlabel("Score")
            ax_bar.set_ylabel("Percentage (%)")
            ax_bar.set_title(f"{label} distribution\nN={total} respondents")

            # Annotations : affiche chaque %
            for idx, pct in enumerate(percentages.values):
                ax_bar.text(idx, pct + 1, f"{pct:.1f}%", ha='center')

    plt.show()

