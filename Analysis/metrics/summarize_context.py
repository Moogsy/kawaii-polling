import pandas as pd
from matplotlib import pyplot as plt

__all__ = ("summarize_context", "plot_context_summary")

def summarize_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule et retourne un DataFrame à une ligne contenant les métriques
    de contexte analytique pour le projet de notations d'images.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les colonnes :
        - 'Category'  (pose-type)
        - 'Model'     (modèle exécutant la pose)
        - 'RaterID'   (identifiant de l'évaluateur)
        - 'Rating'    (critère de notation, ex. 'Kawaii', 'Warmth', 'Expressiveness')
        - 'Score'     (valeur de la note)

    Retours
    -------
    pd.DataFrame
        DataFrame à une seule ligne, colonnes :
        - n_poses                   : nombre de poses-types distinctes
        - n_models                  : nombre de modèles distincts
        - n_evaluators              : nombre d'évaluateurs uniques
        - n_images                  : nombre d'images distinctes (combinaison Category×Model)
        - n_ratings                 : nombre total d'annotations
        - n_criteria                : nombre de critères de notation distincts
        - avg_ratings_per_evaluator : moyenne des annotations par évaluateur
        - avg_ratings_per_image     : moyenne des annotations par image
    """
    # nombre d'images = combinaisons uniques Category × Model
    n_images = df[['Category', 'Model']].drop_duplicates().shape[0]
    n_ratings = len(df)
    n_evaluators = df['RaterID'].nunique()

    counts_by_image = df.groupby(['Category','Model']).size()
    avg_ratings_per_image = counts_by_image.mean()

    counts_by_rater = df.groupby('RaterID').size()
    avg_ratings_per_evaluator = counts_by_rater.mean()


    metrics = {
        'n_poses': df['Category'].nunique(),
        'n_models': df['Model'].nunique(),
        'n_evaluators': n_ratings and n_evaluators or 0,
        'n_images': n_images,
        'n_ratings': n_ratings,
        'n_criteria': df['Rating'].nunique(),
        'avg_ratings_per_evaluator': avg_ratings_per_evaluator,
        'avg_ratings_per_image': avg_ratings_per_image,
    }
    return pd.DataFrame([metrics])


def plot_context_summary(context_df: pd.DataFrame) -> None:
    """
    Affiche les métriques de contexte analytique dans une table matplotlib.

    Paramètres
    ----------
    context_df : pd.DataFrame
        DataFrame à une ligne, avec les colonnes :
        - n_poses
        - n_models
        - n_evaluators
        - n_images
        - n_ratings
        - n_criteria
        - avg_ratings_per_evaluator
        - avg_ratings_per_image
    """
    metrics = context_df.columns.tolist()
    values = context_df.iloc[0].tolist()

    _, ax = plt.subplots(figsize=(6, len(metrics) * 0.4 + 1))
    ax.axis('off')

    table_data = [[metrics[i], values[i]] for i in range(len(metrics))]
    table = ax.table(
        cellText=table_data,
        colLabels=["Métrique", "Valeur"],
        cellLoc='left',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    plt.tight_layout()
    plt.show()

