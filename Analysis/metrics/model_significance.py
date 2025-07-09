import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt

__all__ = ("compute_model_effect_significance", "plot_model_effect_significance")

def compute_model_effect_significance(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Rating, Category), perform one-way ANOVA on scores by Model and compute:
      - F_value
      - p_value
      - neg_log10_p  (=-log10(p_value))
      - significance ("", "*", "**", "***" for p ≤0.05,0.01,0.001)

    H0: All model means are equal.
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
                    F, p = f_oneway(*groups)
                except:
                    F, p = np.nan, np.nan
            else:
                F, p = np.nan, np.nan
            neglogp = -np.log10(p) if (p and p > 0) else np.nan
            if pd.isna(p):
                sig = ""
            elif p <= 0.001:
                sig = "***"
            elif p <= 0.01:
                sig = "**"
            elif p <= 0.05:
                sig = "*"
            else:
                sig = ""
            results.append({
                'Rating': crit,
                'Category': cat,
                'F_value': F,
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
    Plot a heatmap of -log10(p) from ANOVA by Rating × Category,
    annotating significance stars and p-values.

    Main title and subtitle:
      H0: All model means are equal.
      p ≤ 0.05 → reject H0 (model effect); p > 0.05 → no evidence of effect.
    """
    df_sign = compute_model_effect_significance(df_sign)

    # Pivot tables
    pivot_p = df_sign.pivot(index='Rating', columns='Category', values='neg_log10_p')
    pivot_sig = df_sign.pivot(index='Rating', columns='Category', values='significance')
    pivot_pv = df_sign.pivot(index='Rating', columns='Category', values='p_value')

    if vmax is None:
        vmax = np.nanmax(pivot_p.values)

    fig, ax = plt.subplots(constrained_layout=True,
                           figsize=(len(pivot_p.columns) * 0.6, len(pivot_p.index) * 0.6))

    # Main title and subtitle
    fig.suptitle("Model Influence on Perception (ANOVA)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.98,
             "H0: All model means are equal. p ≤ 0.05 → reject H0 (model effect); p > 0.05 → no evidence.",
             ha='center', va='top', fontsize=10)

    # Heatmap
    cax = ax.imshow(pivot_p.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')

    # Ticks
    ax.set_xticks(np.arange(len(pivot_p.columns)))
    ax.set_xticklabels(pivot_p.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(pivot_p.index)))
    ax.set_yticklabels(pivot_p.index)

    # Annotations: stars + p-values
    for i in range(pivot_p.shape[0]):
        for j in range(pivot_p.shape[1]):
            sig = pivot_sig.iat[i, j]
            p = pivot_pv.iat[i, j]
            txt = f"{sig}\n{p:.3f}" if not pd.isna(p) else ""
            ax.text(j, i, txt, ha='center', va='center', color='black', fontsize=8)

    # Colorbar
    fig.colorbar(cax, ax=ax, label='-log10(p-value)')

    plt.show()

