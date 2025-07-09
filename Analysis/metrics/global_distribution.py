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
    Trace par catégorie et critère les barplots (moyenne ± écart-type + médiane/max/min)
    et ajoute un tableau compact des stats globales Likert dans la dernière case.
    """
    # 1) calcul groupé
    grouped = df.groupby(['Category', 'Rating'])
    metrics = grouped['Score'].agg(
        Count='count',
        Average='mean',
        Median='median',
        StdDev=lambda x: x.std(ddof=0),
        Min='min',
        Max='max'
    )

    # 2) extraction pour barplots
    means   = metrics['Average'].unstack('Rating')  # type: ignore
    stds    = metrics['StdDev'   ].unstack('Rating')  # type: ignore
    medians = metrics['Median'   ].unstack('Rating')  # type: ignore
    mins    = metrics['Min'      ].unstack('Rating')  # type: ignore
    maxs    = metrics['Max'      ].unstack('Rating')  # type: ignore

    cats     = means.index.tolist()
    ratings  = means.columns.tolist()
    n_ratings = len(ratings)
    nrows     = int(np.ceil(n_ratings / ncols))
    x         = np.arange(len(cats))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(6*ncols, 4*nrows),
                             squeeze=False)

    cmap = plt.get_cmap('Set3', n_ratings)
    median_color = "Orange"
    max_color    = "Red"
    min_color    = "Blue"

    # Barplots
    for idx, rating in enumerate(ratings):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        color = cmap(idx)
        ax.bar(x, means[rating], width=0.6, color=color)
        ax.errorbar(x, means[rating], yerr=stds[rating],
                    fmt='none', ecolor='gray', capsize=3)
        ax.scatter(x, medians[rating], marker='D',
                   color=median_color, label='Median')
        ax.scatter(x, maxs[rating], marker='^',
                   color=max_color,    label='Max')
        ax.scatter(x, mins[rating], marker='v',
                   color=min_color,    label='Min')
        ax.set_title(rating)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, ha='right')
        ax.set_ylim(0.8, 5.2)
        ax.set_ylabel('Score')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    # 4) Tableau global dans la case vide
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
        if empty_idx == n_ratings:
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
            ax_tab.set_title("Stats globales", pad=10)
        else:
            ax_tab.axis('off')

    # Labels communs
    fig.text(0.5, 0.04, 'Category', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Score', va='center', rotation='vertical', fontsize=14)

    # Légende partagée
    legend_elems = [
        Line2D([0], [0], marker='D', color='w', label='Median',
               markerfacecolor=median_color, markersize=8),
        Line2D([0], [0], marker='^', color='w', label='Max',
               markerfacecolor=max_color, markersize=8),
        Line2D([0], [0], marker='v', color='w', label='Min',
               markerfacecolor=min_color, markersize=8),
    ]
    fig.legend(handles=legend_elems, loc='upper center',
               ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # type: ignore
    plt.show()

    return metrics

