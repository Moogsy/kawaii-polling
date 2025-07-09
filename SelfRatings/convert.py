import pandas as pd

# Fonction principale de conversion du DataFrame au format unifié pour les auto-évaluations
def convert(df: pd.DataFrame) -> pd.DataFrame:

    # Nettoyage de la colonne Category : remplacement de "/" par "-" pour unifier les formats
    df["Category"] = df["Category"].str.replace("/", "-", regex=False)

    # Normalisation des labels dans la colonne Rating
    df["Rating"] = df["Rating"].str.replace("Expressive", "Expressiveness", regex=False)

    # Filtrage : ne conserver que les critères d’intérêt
    valid_ratings = ["Kawaii", "Warmth", "Expressiveness"]
    df = df[df["Rating"].isin(valid_ratings)]  # On garde seulement ces trois types de notations

    # Définition des combinaisons Category/Pose à sélectionner
    selection = {
        "Shy-Gentle": ["C"],
        "Joyful-Smiling": ["E"],
        "Playful-Clumsy": ["A"],
        "Dependant-Needy": ["D"],
        "Escapist-Dreamy": ["B"],
        "Cool-Clever": ["D"],
        "Normal-Warmup": ["C"]
    }

    # Création d’un masque booléen pour filtrer les lignes correspondant aux paires Category/Pose
    mask = pd.Series(False, index=df.index)
    for cat, poses in selection.items():
        mask |= (df["Category"] == cat) & (df["Pose"].isin(poses))

    df = df[mask]  # On applique le filtre

    # Renommage de la colonne "Pose" en "Model" pour correspondre au format cible
    df = df.rename(columns={"Pose": "Model"})

    # Suppression des évaluations faites par P5 (exclu du projet)
    df = df[df["RaterID"] != "P5"]

    # Dictionnaire de correspondance entre codes anonymes et noms réels des modèles
    mapping = dict([
        ("P1", "pragathi"),
        ("P2", "rose"),
        ("P3", "matheusz"),
        ("P4", "victor"),
        #("P5", "jan"),  # volontairement commenté
        ("P6", "leslie"),
        ("P7", "leandre"),
        ("P8", "lorenzo"),
        ("P9", "leandro"),
        ("P10", "renan"),
        ("P11", "kelly"),
        ("P12", "margot"),
        ("P13", "sanne"),
    ])

    # Conversion des identifiants RaterID en noms de modèles
    df["RaterID"] = df["RaterID"].map(mapping)

    # Remplacer la colonne "Model" par les noms de modèles (identiques à RaterID après conversion)
    df["Model"] = df["RaterID"]

    return df


# Fonction d’exécution principale
def main():
    # Chargement du CSV avec index multiples, puis réinitialisation de l’index pour traitement
    df = pd.read_csv("./all_ratings.csv", index_col=[0, 1, 2, 3], low_memory=False).reset_index()

    # Application de la fonction de conversion
    out = convert(df)

    # Enregistrement du tout en csv
    out.to_csv("./all_ratings_new_format.csv")


# Point d’entrée du script
if __name__ == "__main__":
    main()

