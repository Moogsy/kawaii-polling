"""
Tests whether the choice of model has a statistically significant effect on
perception scores for each (rating criterion, pose category) combination.
compute_model_effect_significance() runs a Kruskal-Wallis test across model groups
and returns H statistic, p-value, -log10(p), and significance stars (*, **, ***).
plot_model_effect_significance() visualises the results as a heatmap of -log10(p)
values annotated with significance stars and raw p-values.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal

__all__ = ("compute_model_effect_significance", "plot_model_effect_significance")


def compute_model_effect_significance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque (Rating, Category), effectue un test de Kruskal-Wallis sur les scores par modèle.
    
    Retourne un DataFrame avec :
    - Kruskal H statistic
    - p_value
    - neg_log10_p (=-log10(p))
    - significance ("", "*", "**", "***" pour p ≤ 0.05, 0.01, 0.001)

    H₀ : les distributions des scores des différents modèles sont identiques.
    """
    results = []
    ratings = df['Rating'].unique()
    categories = df['Category'].unique()
    
    for crit in ratings:
        for cat in categories:
            sub = df[(df['Rating'] == crit) & (df['Category'] == cat)]
            groups = [g['Score'].values for _, g in sub.groupby('Model')]
            
            if len(groups) >= 2:
                try:
                    stat, p = kruskal(*groups)
                except:
                    stat, p = np.nan, np.nan
            else:
                stat, p = np.nan, np.nan

            neglogp = -np.log10(p) if (p and p > 0) else np.nan
            sig = (
                "***" if p <= 0.001 else
                "**" if p <= 0.01 else
                "*" if p <= 0.05 else
                ""
            )

            results.append({
                'Rating': crit,
                'Category': cat,
                'Kruskal_H': stat,
                'p_value': p,
                'neg_log10_p': neglogp,
                'significance': sig
            })

    return pd.DataFrame(results)


def plot_model_effect_significance(df_sign: pd.DataFrame,
                                   vmin: float = 0,
                                   vmax: float = None,
                                   cmap: str = "YlOrRd") -> None:
    """
    Trace une heatmap des -log10(p) du test de Kruskal-Wallis par Rating × Category.
    Affiche en annotation les étoiles de significativité et les valeurs de p.
    """
    df_sign = compute_model_effect_significance(df_sign)

    pivot_p = df_sign.pivot(index='Rating', columns='Category', values='neg_log10_p')
    pivot_sig = df_sign.pivot(index='Rating', columns='Category', values='significance')
    pivot_pv = df_sign.pivot(index='Rating', columns='Category', values='p_value')

    if vmax is None:
        vmax = np.nanmax(pivot_p.values)

    fig, ax = plt.subplots(constrained_layout=True,
                           figsize=(len(pivot_p.columns) * 0.6, len(pivot_p.index) * 0.6))

    # Titre principal
    fig.suptitle("Model Influence on Perception (Kruskal-Wallis Test)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.98,
             "H₀: All model distributions are equal. "
             "p ≤ 0.05 → reject H₀ (model effect); p > 0.05 → no evidence.",
             ha='center', va='top', fontsize=10)

    # Heatmap principale
    cax = ax.imshow(pivot_p.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')

    # Ticks
    ax.set_xticks(np.arange(len(pivot_p.columns)))
    ax.set_xticklabels(pivot_p.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(pivot_p.index)))
    ax.set_yticklabels(pivot_p.index)

    # Annotations texte : étoiles et p-valeurs
    for i in range(pivot_p.shape[0]):
        for j in range(pivot_p.shape[1]):
            sig = pivot_sig.iat[i, j]
            p = pivot_pv.iat[i, j]
            txt = f"{sig}\n{p:.3f}" if not pd.isna(p) else ""
            ax.text(j, i, txt, ha='center', va='center', color='black', fontsize=8)

    # Barre de couleur
    fig.colorbar(cax, ax=ax, label='-log10(p-value)')

    plt.show()

