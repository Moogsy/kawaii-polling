from pathlib import Path
import pandas as pd
import metrics

def load_df_from_csv(csv_paths: list[Path]) -> pd.DataFrame:
    """
    Charge et concatène les CSV de notations en un DataFrame unique.
    """
    dfs = [
        pd.read_csv(path, index_col=[0,1,2,3], low_memory=False)
        for path in csv_paths
    ]
    return pd.concat(dfs).reset_index()

def run_part1_context(df: pd.DataFrame) -> None:
    """
    Partie I. Présentation du protocole
    Question : Quelle est la taille et la structure de notre jeu de données ?
    """
    ctx = metrics.summarize_context(df)
    metrics.plot_context_summary(ctx)

def run_part2_global_stats(df: pd.DataFrame) -> None:
    """
    Partie II. Statistiques descriptives globales
    Question : Comment se répartissent les scores (Kawaii, Warmth, Expressiveness) ?
    """
    metrics.plot_metrics_per_category(df)

def run_part3_correlations(df: pd.DataFrame) -> None:
    """
    Partie III. Corrélations entre critères
    Question : Dans quelle mesure Kawaii, Warmth et Expressiveness sont-ils corrélés ?
    """
    metrics.plot_pairwise_correlation_scatter(df, level="rater")
    metrics.plot_pairwise_correlation_scatter(df, level="image")

def run_part4_inter_category(df: pd.DataFrame, images_folder: Path) -> None:
    """
    Partie IV. Comparaison inter-catégorie
    Question : Quelles poses-types performent le mieux sur chaque critère et quels sont leurs représentants ?
    """
    metrics.plot_multicriteria_category_representatives(df, base_dir=images_folder)

def run_part5_intra_category(df: pd.DataFrame, images_folder: Path) -> None:
    """
    Partie V. Analyse intra-catégorie
    1) Divisiveness : Quelles images sont les plus vs. les moins clivantes ?
    2) Model effect : La morphologie du modèle influence-t-elle la perception ?
    """
    # 5.1 Polarisation des évaluations
    metrics.plot_divisiveness_per_rating_percentage(df, images_folder=images_folder)
    # 5.2 Effet morphologique (ANOVA)
    metrics.plot_model_effect_significance(df)

def run_part6_perception(df: pd.DataFrame) -> None:
    """
    Partie VI. Analyse de la perception : biais, consistances, polarités
    Pour chaque critère (Kawaii, Warmth, Expressiveness) :
      - Calcul du biais de notation par évaluateur
      - Affichage des statistiques de biais (mean, % extrêmes)
      - Calcul de la consistance inter-évaluateurs (mean Spearman ρ)
      - Affichage de la distribution de consistance
    """
    metrics.plot_rater_bias_stats(df)


def main():
    project_root = Path(__file__).parent.parent
    images_folder = project_root / "static"
    self_rating = project_root / "SelfRatings" / "all_ratings_new_format.csv"
    external_ratings = list((project_root / "Ratings").glob("*.csv"))
    all_ratings = external_ratings + [self_rating]

    df = load_df_from_csv(all_ratings)

    df = df[~df["Model"].str.lower().eq("kelly")]

    # Exécute les analyses section par section
    #run_part1_context(df)
    #run_part2_global_stats(df)
    run_part3_correlations(df)
    #run_part4_inter_category(df, images_folder)
    #run_part5_intra_category(df, images_folder)
    #run_part6_perception(df)

if __name__ == "__main__":
    main()

