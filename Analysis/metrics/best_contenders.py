"""
Identifies the best, worst, and "typical" model for every (criterion, category) pair
and renders a summary grid.  For each cell the grid shows the category mean score
alongside three thumbnail images — best, typical (closest to mean), and worst model —
each annotated with its per-model average.  Criteria columns are ordered by the first
criterion so the overall ranking is immediately visible.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

__all__ = ("plot_multicriteria_category_representatives",)

def compute_category_ranking(df: pd.DataFrame, rating: str) -> pd.DataFrame:
    """
    For a given rating criterion, compute per category:
      - mean, std, n
      - best_model & its mean
      - worst_model & its mean
      - typical_model (closest to category mean) & its mean
    """
    df_r = df[df["Rating"] == rating]
    # overall stats per category
    cat_stats = (
        df_r.groupby("Category")["Score"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
    )
    # model means
    model_stats = (
        df_r.groupby(["Category","Model"])["Score"]
            .mean()
            .reset_index(name="model_mean")
    )
    # best and worst
    idx_best  = model_stats.groupby("Category")["model_mean"].idxmax()
    idx_worst = model_stats.groupby("Category")["model_mean"].idxmin()
    best = model_stats.loc[idx_best].rename(
        columns={"Model":"best_model","model_mean":"best_model_mean"}
    )
    worst = model_stats.loc[idx_worst].rename(
        columns={"Model":"worst_model","model_mean":"worst_model_mean"}
    )
    # typical: closest to category mean
    cat_mean = cat_stats.set_index("Category")["mean"]
    df_comb = model_stats.set_index("Category").join(cat_mean, on="Category")
    df_comb = df_comb.reset_index()
    df_comb["diff"] = (df_comb["model_mean"] - df_comb["mean"]).abs()
    idx_typical = df_comb.groupby("Category")["diff"].idxmin()
    typical = df_comb.loc[idx_typical].rename(
        columns={"Model":"typical_model","model_mean":"typical_model_mean"}
    )[
        ["Category","typical_model","typical_model_mean"]
    ]
    # merge all
    return (
        cat_stats
        .merge(best,     on="Category")
        .merge(worst,    on="Category")
        .merge(typical,  on="Category")
    )

def compute_image_path(entry: tuple[str, str], base_dir: str = "../static") -> str:
    """
    Build path: ../static/<Category>/blurred_<Model>.png
    """
    category, model = entry
    safe_model = "".join(c if c.isalnum() else "_" for c in model)
    return os.path.join(base_dir, category, f"blurred_{safe_model}.png")

def plot_multicriteria_category_representatives(
    df: pd.DataFrame,
    criteria: list[str] = ["Kawaii","Warmth","Expressiveness"],
    base_dir: Path | str = "../static",
    title: str = "Pose Categories: Best, Typical & Worst Representatives"
) -> None:
    """
    Display a grid:
     - Row 0: title
     - Row 1: criterion labels (one meta-column per criterion)
     - Rows 2..: for each category (ordered by first criterion desc):
         * even col: category name + mean score
         * odd  col: three inset images (best, typical, worst) each labeled model_avg=…
    """
    # master order by first criterion
    master = compute_category_ranking(df, criteria[0]).sort_values("mean", ascending=False)
    order = master["Category"].tolist()

    # compute rankings for each criterion in master order
    rankings = {}
    for crit in criteria:
        r = compute_category_ranking(df, crit).set_index("Category").reindex(order).reset_index()
        rankings[crit] = r

    n_cats = len(order)
    ncrit  = len(criteria)
    ncols  = 2 * ncrit

    # GridSpec: 1 title row, 1 label row, n_cats content rows
    height_ratios = [0.6, 0.4] + [1]*n_cats
    fig = plt.figure(constrained_layout=True,
                     figsize=(4*ncrit, 2*(n_cats+1)))
    gs = GridSpec(n_cats+2, ncols, figure=fig, height_ratios=height_ratios)

    # Title row
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")
    ax_t.text(0.5, 0.5, title, ha="center", va="center",
              fontsize=16, fontweight="bold")

    # Criterion labels
    for j, crit in enumerate(criteria):
        ax_l = fig.add_subplot(gs[1, j*2:(j*2+2)])
        ax_l.axis("off")
        ax_l.text(0.5, 0.5, crit, ha="center", va="center",
                  fontsize=14, fontweight="semibold")

    # Content rows
    for j, crit in enumerate(criteria):
        r = rankings[crit]
        for i, row in r.iterrows():
            ridx = i + 2  # content row index

            # Category + mean
            ax_txt = fig.add_subplot(gs[ridx, j*2])
            ax_txt.axis("off")
            ax_txt.text(0.05, 0.6, row["Category"], ha="left", va="center", fontsize=10)
            ax_txt.text(0.05, 0.2, f"μ={row['mean']:.2f}", ha="left", va="center", fontsize=8)

            # Best / Typical / Worst insets
            ax_img = fig.add_subplot(gs[ridx, j*2+1])
            ax_img.axis("off")

            for loc, key in zip(
                ["upper left", "upper center", "upper right"],
                [("best_model","best_model_mean"),
                 ("typical_model","typical_model_mean"),
                 ("worst_model","worst_model_mean")]
            ):
                model, mmean = row[key[0]], row[key[1]]
                path = compute_image_path((row["Category"], model), base_dir)
                iax = inset_axes(ax_img, width="30%", height="90%", loc=loc, borderpad=0.1)
                iax.imshow(plt.imread(path)); iax.axis("off")
                iax.text(0.5, -0.05, f"model_avg={mmean:.2f}",
                         ha="center", va="top", transform=iax.transAxes, fontsize=7)

    plt.show()

