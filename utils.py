# General
import logging
import pickle
from itertools import combinations
from typing import Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import anderson, kstest, kurtosis, shapiro, skew
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

RANDOM_STATE = 0
TEST_SIZE = 0.2
ALPHA = 0.02
TARGET = "Exited"


# Hypothesis testing
def check_hypothesis(test, alpha: float = 0.02) -> str:
    logging.info("Checking hypothesis with alpha = %s", alpha)
    if test.pvalue < alpha:
        return "Nous avons suffisamment d´évidence pour rejeter l´hypothèse nulle"
    else:
        return "Nous n'avons pas suffisamment d'évidence pour rejeter l'hypothèse nulle"


def check_normality(data, alpha=0.02):
    """
    Check if the data is normally distributed using multiple tests.

    Parameters:
    data (array-like): The data to be tested.
    alpha (float): The significance level. Default is 0.05.

    Returns:
    dict: A dictionary with the results of the normality tests.
    """
    results = {}

    # Shapiro-Wilk Test
    if len(data) <= 5000:
        stat, p_value = shapiro(data)
        results["shapiro"] = {
            "statistic": stat,
            "p_value": p_value,
            "normal": p_value > alpha,
        }
        logging.info(f"Shapiro-Wilk Test Statistic: {stat}, P-value: {p_value}")
    else:
        logging.info("Shapiro-Wilk Test skipped due to large sample size.")

    # Kolmogorov-Smirnov Test
    stat, p_value = kstest(data, "norm", args=(np.mean(data), np.std(data)))
    results["kolmogorov_smirnov"] = {
        "statistic": stat,
        "p_value": p_value,
        "normal": p_value > alpha,
    }
    logging.info(f"Kolmogorov-Smirnov Test Statistic: {stat}, P-value: {p_value}")

    # Anderson-Darling Test
    result = anderson(data, dist="norm")
    results["anderson_darling"] = {
        "statistic": result.statistic,
        "critical_values": result.critical_values,
        "significance_levels": result.significance_level,
        "normal": result.statistic
        < result.critical_values[2],  # Using 5% significance level
    }
    logging.info(
        f"Anderson-Darling Test Statistic: {result.statistic}, Critical Values: {result.critical_values}"
    )

    # Visual inspection using subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 3))

    # Histogram with KDE
    sns.histplot(data, kde=True, stat="density", linewidth=0, ax=axes[0])
    axes[0].set_title("Histogram avec KDE")

    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title("Représentation Q-Q")

    plt.tight_layout()
    plt.show()

    # Interpret the results
    if p_value > alpha:
        print("The data is normally distributed (fail to reject H0).")
        return True
    else:
        print("The data is not normally distributed (reject H0).")
        return False


def describe_distribution(data: pd.DataFrame, name: str, hue=None, xlimit=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    plt.suptitle(f"Répartition de la variable '{name}'")
    h = sns.histplot(data=data, x=name, hue=hue, ax=ax1)
    b = sns.boxplot(data=data, x=name, hue=hue, ax=ax2)
    if xlimit is not None:
        ax1.set_xlim(*xlimit)
        ax2.set_xlim(*xlimit)
    plt.tight_layout()
    plt.show()

    # Describe the distribution
    desc = data[name].describe()
    skewness = skew(data[name])
    kurt = kurtosis(data[name])

    print(desc)
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurt}")


def sauvegarder_model(
    model,
    file_name: str = "../data/models/best-model",
    timestamp=None,
    only_latest: bool = False,
):
    if not only_latest:
        with open(f"{file_name}-{str(timestamp)}.pkl", "wb") as f:
            pickle.dump(model, f)
    with open(f"{file_name}-latest.pkl", "wb") as f:
        pickle.dump(model, f)


def ouvrir_model(file_name: str = "../data/models/best-model-latest.pkl"):
    with open(file_name, "rb") as f:
        clf = pickle.load(f)
    return clf


def identifier_iqr_outlier(df: pd.DataFrame, columns=None, factor=1.5):
    """
    Identifie les lignes outliers selon IQR sur colonnes sélectionnées
    X : DataFrame pandas
    y : Series ou array
    columns : liste des colonnes à analyser (toutes si None)
    factor : multiplicateur pour l'IQR
    """
    # Si aucune colonne spécifiée, utiliser toutes
    if columns is None:
        columns = df.columns

    # Copie pour éviter de modifier les originaux
    df = df.copy()

    # Liste des index à supprimer
    out_indexlist = []

    for col in columns:
        Q1 = np.nanpercentile(df[col], 25.0)
        Q3 = np.nanpercentile(df[col], 75.0)
        cut_off = (Q3 - Q1) * factor
        lower, upper = Q1 - cut_off, Q3 + cut_off

        outliers_index = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
        out_indexlist.extend(outliers_index)

    # Supprimer les doublons
    out_indexlist = set(out_indexlist)

    # Garder seulement les lignes "clean"
    clean_idx = df.index.difference(out_indexlist)

    return df.loc[clean_idx], out_indexlist


def _make_label(items: list) -> str:
    """
    Retourne un libellé lisible pour un bloc de modalités.
    - Pour des nombres consécutifs :  [1,2,3]  -> '1-3'
    - Pour des strings : ['France','Spain']   -> 'France+Spain'
    """
    items = sorted(items)
    # cas numérique entier consécutif
    if all(isinstance(x, (int, np.integer)) for x in items):
        if len(items) == 1:
            return str(items[0])
        if max(items) - min(items) == len(items) - 1:
            return f"{min(items)}-{max(items)}"
    # sinon (strings ou non consécutif) : concaténation
    return "+".join(map(str, items))


def trouver_groupes(
    data: pd.DataFrame,
    nom_colonne: str,
    alpha: float = ALPHA,
    ordinal: bool | None = None,
) -> dict:
    """
    Détermine, par tests de proportions, quelles modalités de `nom_colonne`
    peuvent être fusionnées. Renvoie un mapping
    {modalité d'origine -> libellé de bucket (string)}.

    - Si `ordinal=True`   : ne compare que les modalités adjacentes
                            (utile pour Tenure 0-10).
    - Si `ordinal=False`  : compare toutes les paires (utile pour Geography).
    - Si `ordinal=None`   : détecte automatiquement
          -> numérique entier compact       -> ordinal
          -> sinon                           -> non ordinal
    """
    contingency = pd.crosstab(data[nom_colonne], data[TARGET])
    levels = list(contingency.index)

    # ── Choix des paires à comparer ───────────────────────────
    if ordinal is None:
        # heuristique : colonne de type int & valeurs continues => ordinal
        ordinal = pd.api.types.is_integer_dtype(contingency.index) and (
            max(levels) - min(levels) + 1 == len(levels)
        )

    if ordinal:
        levels_sorted = sorted(levels)
        pairs = [
            (levels_sorted[i], levels_sorted[i + 1])
            for i in range(len(levels_sorted) - 1)
        ]
    else:
        pairs = list(combinations(levels, 2))

    # ── P-values brutes ───────────────────────────────────────
    p_vals = []
    for a, b in pairs:
        succ = [contingency.loc[a, 1], contingency.loc[b, 1]]
        tot = [contingency.loc[a].sum(), contingency.loc[b].sum()]
        _, p = proportions_ztest(succ, tot, alternative="two-sided")
        p_vals.append(p)

    # ── Correction Holm (contrôle du risque global) ───────────
    _, p_corr, _, _ = multipletests(p_vals, method="holm")

    # ── Fusion des blocs ──────────────────────────────────────
    blocks = [[l] for l in levels]  # départ : chaque valeur dans son bloc

    def find_block(x):
        return next(i for i, b in enumerate(blocks) if x in b)

    for (a, b), p in zip(pairs, p_corr):
        if p >= alpha:  # « pas différent » → fusion
            i, j = find_block(a), find_block(b)
            if i != j:
                blocks[i].extend(blocks[j])
                blocks.pop(j)

    # ── Construction du mapping ───────────────────────────────
    bucket_map = {}
    for bloc in blocks:
        label = _make_label(bloc)
        for val in bloc:
            bucket_map[val] = label
    return bucket_map


def regrouper(
    data: pd.DataFrame, bucket_map: dict, column_name: str, new_column: str
) -> pd.DataFrame:
    """
    Ajoute une colonne catégorielle ordonnée (ordre des buckets
    = ordre « naturel » des modalités) au DataFrame.
    """
    ordered_labels = list(dict.fromkeys(bucket_map[v] for v in sorted(bucket_map)))
    return data.assign(
        **{
            new_column: (
                data[column_name]
                .map(bucket_map)
                .astype(pd.CategoricalDtype(categories=ordered_labels, ordered=True))
            )
        }
    )


def trouver_continuous_bucket(
    df: pd.DataFrame,
    column_name: str = "Age",
    col_target: str = "Exited",
    min_per_bin: int = 400,
    alpha: float = 0.05,
):
    import numpy as np
    import pandas as pd
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.proportion import proportions_ztest

    df = df.copy()
    df[column_name] = df[column_name].round().astype(int)

    counts = df[column_name].value_counts().sort_index()
    bins = [[val] for val in counts.index]

    # ── pré-fusion (taille mini)
    i = 0
    while i < len(bins):
        if counts.loc[bins[i]].sum() < min_per_bin and i < len(bins) - 1:
            bins[i].extend(bins[i + 1])
            bins.pop(i + 1)
        else:
            i += 1

    # ── boucle tant qu’il existe au moins une paire « adjacente non signif. »
    while True:
        # 1. mapping âge → bloc_id
        age2bloc = {age: idx for idx, bloc in enumerate(bins) for age in bloc}
        df["_blk"] = df[column_name].map(age2bloc)

        # 2. table et couples vraiment adjacents
        tab = pd.crosstab(df["_blk"], df[col_target])
        idx = sorted(tab.index)
        pairs, pvals = [], []
        for a, b in zip(idx[:-1], idx[1:]):
            succ = [tab.loc[a, 1], tab.loc[b, 1]]
            tot = [tab.loc[a].sum(), tab.loc[b].sum()]
            _, p = proportions_ztest(succ, tot)
            pairs.append((a, b))
            pvals.append(p)

        # 3. Holm
        if not pvals:
            break
        _, p_adj, _, _ = multipletests(pvals, method="holm")

        # 4. cherche la 1re paire non signif. ; sinon stop
        try:
            a, b = next((ab for ab, p in zip(pairs, p_adj) if p >= alpha))
        except StopIteration:
            break  # plus rien à fusionner

        # 5. fusion des deux blocs
        bins[a].extend(bins[b])
        bins.pop(b)

    # ── mapping final âge → libellé
    def label(bloc):
        bloc = sorted(bloc)
        return f"{bloc[0]}" if bloc[0] == bloc[-1] else f"{bloc[0]}-{bloc[-1]}"

    bucket_map = {age: label(bloc) for bloc in bins for age in bloc}
    return bucket_map


def apply_buckets(
    df: pd.DataFrame,
    bucket_map: Dict[int, str],
    col_age: str = "Age",
    new_col: str = "AgeBucket",
) -> pd.DataFrame:
    """Ajoute la colonne catégorielle ordonnée."""
    ordered = list(dict.fromkeys(bucket_map[a] for a in sorted(bucket_map)))
    return df.assign(
        **{
            new_col: (
                df[col_age]
                .round()
                .astype(int)
                .map(bucket_map)
                .astype(pd.CategoricalDtype(categories=ordered, ordered=True))
            )
        }
    )


def influence_variable(score_chi2, table_contingency):
    cramer_v = np.sqrt(
        score_chi2
        / (table_contingency.values.sum() * (min(table_contingency.shape) - 1))
    )
    if cramer_v < 0.1:
        print("\nInfluence de la variable sur le target: Faible")
    elif cramer_v < 0.3:
        print("\nInfluence de la variable sur le target: Légère")
    else:
        print("\nInfluence de la variable sur le target: Moyenne")
