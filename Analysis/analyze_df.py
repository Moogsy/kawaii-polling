from pathlib import Path
import pandas as pd
import metrics


def load_df_from_csv(csv_path: list[Path]) -> pd.DataFrame:

    df_list = [
            pd.read_csv(path, index_col=[0, 1, 2, 3], low_memory=False) 
            for path in csv_path
    ]

    big_df = pd.concat(df_list)
    return big_df.reset_index() # Flatten the dataframe for easier access


if __name__ == "__main__":
    df = load_df_from_csv([*Path("../Ratings").glob("*.csv")])
    metrics.plot_context_summary(metrics.summarize_context(df))
    #metrics.plot_metrics_per_category(df)
    #metrics.plot_dual_correlation_matrices(df, method="spearman")
    #metrics.plot_divisiveness_per_rating_percentage(df)
    #metrics.plot_multicriteria_category_representatives(df)
    #metrics.plot_model_effect_significance(df)

