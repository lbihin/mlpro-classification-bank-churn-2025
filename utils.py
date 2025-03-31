# General
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import skew, kurtosis
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import anderson, kstest, shapiro

RANDOM_STATE = 0
TEST_SIZE = 0.2


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


def describe_distribution(data: pd.DataFrame, name: str):
    fig, ax = plt.subplots(2, 1, figsize=(12, 5))
    plt.suptitle(f"Répartition de la variable '{name}'")
    sns.histplot(data=data, x=name, ax=ax[0])
    sns.boxplot(data=data, x=name, ax=ax[1])
    plt.tight_layout()
    plt.show()

    # Describe the distribution
    desc = data[name].describe()
    skewness = skew(data[name])
    kurt = kurtosis(data[name])

    print(desc)
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurt}")
