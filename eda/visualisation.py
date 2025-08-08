from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def afficher_feature_vs_target(
    column: str,
    data: pd.DataFrame,
    kind: Literal["continuous", "discrete"] = "discrete",
):

    if kind == "discrete":

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        # Non-normalized count plot
        sns.countplot(data=data, x=column, hue="Exited", ax=ax1)
        ax1.set_title(f"Dénombrement")
        ax1.set_ylabel("Count")
        ax1.legend(
            title="Résiliation",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            labels=["Non", "Oui"],
        )

        # Normalized proportional plot
        crosstab = pd.crosstab(data[column], data.Exited, normalize="index")
        crosstab.plot(kind="bar", stacked=True, ax=ax2)
        ax2.set_title(f"Normalisé")
        ax2.set_ylabel("Proportion")
        ax2.legend(
            title="Résiliation",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            labels=["Non", "Oui"],
        )

    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        sns.histplot(data, x=column, hue="Exited", kde=True, ax=ax1)
        ax1.legend(
            title="Résiliation",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            labels=["Non", "Oui"],
        )
        sns.boxplot(data=data, x=column, hue="Exited", ax=ax2)
        ax2.legend(
            title="Résiliation",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            labels=["Non", "Oui"],
        )

    # Ajouter un titre général
    fig.suptitle(f"Répartition de {column}", fontsize=14)
    plt.tight_layout()
    plt.show()
