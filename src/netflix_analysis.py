
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(filepath):
    """Charge le dataset Netflix."""
    return pd.read_csv(filepath)


def prepare_data(df):
    """Nettoie et prépare les données Netflix."""

    df = df.copy()

    # Correction des durées enregistrées par erreur dans rating
    erreur_duration = df["rating"].str.contains(
        "min",
        na=False
    )

    df.loc[erreur_duration, "duration"] = (
        df.loc[erreur_duration, "rating"]
    )

    df.loc[erreur_duration, "rating"] = pd.NA

    # Conversion de date_added
    df["date_added"] = pd.to_datetime(
        df["date_added"],
        format="mixed",
        errors="coerce"
    )

    # Création de l'année d'ajout
    df["year_added"] = df["date_added"].dt.year

    return df


def count_content_types(df):
    """Compte les contenus par type."""

    return df["type"].value_counts()


def contents_added_by_year(df):
    """Compte les contenus ajoutés par année et par type."""

    return (
        df
        .groupby(["year_added", "type"])
        .size()
        .reset_index(name="nombre_contenus")
    )


def prepare_movie_durations(df):
    """Prépare la durée des films en minutes."""

    movies = df.loc[
        df["type"] == "Movie"
    ].copy()

    movies["duration_min"] = (
        movies["duration"]
        .str.extract(r"(\d+)")
        .astype(float)
    )

    return movies


def get_longest_movie(df):
    """Retourne le film le plus long."""

    movies = prepare_movie_durations(df)

    index_longest = movies["duration_min"].idxmax()

    return movies.loc[
        index_longest,
        ["title", "duration", "release_year"]
    ]


def prepare_tv_shows(df):
    """Prépare les données relatives aux séries."""

    tv_shows = df.loc[
        df["type"] == "TV Show"
    ].copy()

    tv_shows["nombre_saisons"] = (
        tv_shows["duration"]
        .str.extract(r"(\d+)")
        .astype(float)
    )

    return tv_shows


def count_shows_by_seasons(df):
    """Compte les séries selon leur nombre de saisons."""

    tv_shows = prepare_tv_shows(df)

    return (
        tv_shows["nombre_saisons"]
        .value_counts()
        .sort_index()
    )


def get_top_categories(df, n=10):
    """Retourne les catégories les plus représentées."""

    categories = (
        df["listed_in"]
        .dropna()
        .str.split(", ")
        .explode()
    )

    return categories.value_counts().head(n)


def get_top_directors(df, n=10):
    """Retourne les réalisateurs les plus présents."""

    directors = (
        df["director"]
        .dropna()
        .str.split(", ")
        .explode()
    )

    return directors.value_counts().head(n)


def get_director_actors(df, director_name, n=5):
    """Retourne les acteurs les plus fréquents d'un réalisateur."""

    director_data = df.loc[
        df["director"].notna()
        & df["director"].str.contains(
            director_name,
            case=False,
            na=False
        )
    ]

    actors = (
        director_data["cast"]
        .dropna()
        .str.split(", ")
        .explode()
    )

    return actors.value_counts().head(n)
