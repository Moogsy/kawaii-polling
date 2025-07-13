import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def compute_global_stats_likert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule des stats descriptives adaptées à une échelle de Likert pour chaque critère.

    Retour:
    - mean, median, mode, Q1, Q3, IQR, skew, pct≥4 (% de réponses 4 ou 5), pct≤2 (% de réponses 1 ou 2)
    """
    def mode_func(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else np.nan

    stats = (
        df.groupby('Rating')['Score']
          .agg(
              mean='mean',
              median='median',
              mode=mode_func,
              q1=lambda x: x.quantile(0.25),
              q3=lambda x: x.quantile(0.75),
              iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
              skew='skew',
              pct_top2=lambda x: (x >= 4).mean(),
              pct_bottom2=lambda x: (x <= 2).mean(),
          )
          .reset_index()
    )
    stats['pct_top2'] = (stats['pct_top2'] * 100).round(3)
    stats['pct_bottom2'] = (stats['pct_bottom2'] * 100).round(3)
    return stats.round(3)

def plot_metrics_per_category(df: pd.DataFrame, ncols: int = 2):
    """
    Trace des boxplots par catégorie et par critère (Rating) pour des scores sur échelle de Likert.
    Corrige les warnings liés à l'utilisation de `palette` sans `hue` et à `set_ticklabels`.
    Affiche ≈ X/model sous les catégories + ajoute une ligne de médiane globale + légende complète.
    """
    import seaborn as sns
    from matplotlib.lines import Line2D

    ratings = df['Rating'].unique()
    n_ratings = len(ratings)
    nrows = int(np.ceil(n_ratings / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    global_medians = df.groupby("Rating")["Score"].median().to_dict()

    for idx, rating in enumerate(ratings):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        data_subset = df[df['Rating'] == rating]

        # Comptage par catégorie uniquement pour le Rating courant
        grouped = data_subset.groupby("Category")
        categories_ordered = sorted(grouped.groups.keys())
        categories_with_n = []

        for cat in categories_ordered:
            sub = data_subset[data_subset["Category"] == cat]
            n_models = sub["Model"].nunique()
            n_scores = sub.shape[0]
            avg_per_model = round(n_scores / n_models, 1) if n_models > 0 else 0
            label = f"{cat}\nµ ≈ {avg_per_model}"
            categories_with_n.append(label)

        sns.boxplot(
            data=data_subset, x='Category', y='Score', ax=ax,
            hue='Category', legend=False,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
            flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
            order=categories_ordered
        )
        ax.axhline(global_medians[rating], color='gray', linestyle='--', linewidth=1, label='Global Median')
        ax.set_title(f"Distribution of '{rating}' Scores by Pose Category", fontsize=12)
        ax.set_xticks(range(len(categories_with_n)))
        ax.set_xticklabels(categories_with_n, rotation=45, ha='right')
        ax.set_ylim(0.8, 5.2)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xlabel("Pose Category")
        ax.set_ylabel("Likert Score")

    # Ajouter le tableau des stats globales dans la case vide suivante
    stats_global = compute_global_stats_likert(df)
    stats_global = stats_global.rename(columns={
        'pct_top2': 'pct≥4',
        'pct_bottom2': 'pct≤2',
        'q1': 'Q1',
        'q3': 'Q3'
    })

    for empty_idx in range(n_ratings, nrows * ncols):
        row, col = divmod(empty_idx, ncols)
        ax_tab = axes[row][col]
        ax_tab.axis('off')
        tbl = ax_tab.table(
            cellText=stats_global.values.tolist(),
            colLabels=stats_global.columns.tolist(),
            cellLoc='center',
            colLoc='center',
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.2)
        for (r, _), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(ha='left')
        ax_tab.set_title("Global Likert Scale Statistics", pad=10)

    # Légende commune
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Mean',
               markerfacecolor='black', markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Outlier',
               markerfacecolor='white', markeredgecolor='black', markersize=8),
        Line2D([0], [0], color='gray', linestyle='--', label='Global Median'),
        Line2D([], [], color='none', label='µ ≈ average ratings per model')
    ]
    fig.legend(handles=legend_elems, loc='upper center', ncol=4, frameon=False)

    # Titre principal
    fig.suptitle("Distribution of Likert Ratings Across Pose Categories and Perceptual Criteria",
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

