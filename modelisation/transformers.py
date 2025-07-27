import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score


class ChurnProbabilityScore(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        threshold=0.5,
        countries="all",
        optimize_threshold=False,
        optimizer="Nelder-Mead",
        verbose=False,
    ):
        """
        threshold : seuil utilisé pour calculer le F1 pendant l'optimisation (0.5 par défaut)
        countries : 'all' ou liste de pays à traiter
        optimize_threshold : si True, cherche le meilleur seuil par pays après optimisation des poids
        optimizer : méthode scipy.optimize (Nelder-Mead ou Powell recommandé)
        verbose : affiche la progression lors du fit
        """
        self.threshold = threshold
        self.countries = countries
        self.optimize_threshold = optimize_threshold
        self.optimizer = optimizer
        self.verbose = verbose
        self.weights_by_country_ = {}
        self.threshold_by_country_ = {}

    def _compute_score(self, X, weights, country):
        """Calcule les probabilités de churn pour tout X avec des poids donnés."""
        w_age, w_num_products, w_active, w_has_balance, w_geography = weights

        has_balance = (X["Balance"] > 0).astype(int)
        geography_match = (X["Geography"].astype(str) == country).astype(int)
        age_norm = (X["Age"] - 18) / (X["Age"].max() - 18)

        score = (
            w_age * age_norm
            + w_num_products * X["NumOfProducts"]
            + w_active * X["IsActiveMember"]
            + w_has_balance * has_balance
            + w_geography * geography_match
        )

        return 1 / (1 + np.exp(-score))

    def _objective(self, weights, X, y, country):
        """Objectif : maximiser F1 (minimiser -F1) pour un pays donné."""
        prob = self._compute_score(X, weights, country)
        preds = (prob >= self.threshold).astype(int)
        return -f1_score(y, preds)

    def _find_best_threshold(self, prob, y):
        """Cherche le meilleur seuil F1 sur la probabilité donnée."""
        thresholds = np.linspace(0.1, 0.9, 17)  # 0.1 → 0.9 par pas de 0.05
        best_f1, best_thresh = 0, 0.5
        for t in thresholds:
            f1 = f1_score(y, (prob >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
        return best_thresh

    def fit(self, X, y):
        # Convert Geography en string pour éviter bugs de type
        X = X.copy()
        X["Geography"] = X["Geography"].astype(str)

        # Déterminer pays à traiter
        countries_to_fit = (
            X["Geography"].unique() if self.countries == "all" else self.countries
        )

        for country in countries_to_fit:
            mask = X["Geography"] == country
            X_country, y_country = X[mask], y[mask]

            if len(X_country) == 0:
                if self.verbose:
                    print(
                        f"[WARN] Aucun échantillon pour {country}, poids par défaut appliqués."
                    )
                self.weights_by_country_[country] = [0, 0, 0, 0, 0]
                self.threshold_by_country_[country] = 0.5
                continue

            # Initialisation et optimisation des poids
            init_weights = [0.4, -0.6, -0.5, 0.3, 0.2]
            res = minimize(
                self._objective,
                init_weights,
                args=(X_country, y_country, country),
                method=self.optimizer,
                options={"maxiter": 200},
            )

            self.weights_by_country_[country] = res.x

            # Optimiser le seuil F1 si demandé
            if self.optimize_threshold:
                prob = self._compute_score(X_country, res.x, country)
                best_thresh = self._find_best_threshold(prob, y_country)
                self.threshold_by_country_[country] = best_thresh
            else:
                self.threshold_by_country_[country] = self.threshold

            if self.verbose:
                print(
                    f"[{country}] Weights={res.x}, Threshold={self.threshold_by_country_[country]:.2f}"
                )

        return self

    def transform(self, X):
        if not self.weights_by_country_:
            raise ValueError("Le transformeur doit être fit() avant transform().")

        X_out = X.copy()
        X_out["Geography"] = X_out["Geography"].astype(str)
        churn_scores = np.zeros(len(X_out))

        # Vectorisation : calcul score pays par pays
        for country, weights in self.weights_by_country_.items():
            mask = X_out["Geography"] == country
            churn_scores[mask] = self._compute_score(X_out[mask], weights, country)

        return pd.DataFrame(data={"churn_score": churn_scores}, index=X_out.index)
        X_out["churn_score"] = churn_scores
        return X_out


class GenerateBinaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        percentiles=[25, 50, 75],
        young_high_credit_score_pairs=[(20, 700), (35, 700), (50, 700)],
    ):
        self.percentiles = percentiles
        self.young_high_credit_score_pairs = young_high_credit_score_pairs
        self.ages = [20, 35, 50]

    def fit(self, X, y=None):
        # Calculer les percentiles pour CreditScore
        self.credit_score_percentile = np.percentile(X["CreditScore"], self.percentiles)
        self.balance_percentile = np.percentile(X["Balance"], self.percentiles)
        self.num_products = X.NumOfProducts.unique()
        return self

    def transform(self, X, y=None):
        new_features = {}
        # Générer les colonnes binaires pour chaque percentile
        new_features.update(
            {
                f"IsZeroBalance_{percentile}pCreditScore": (
                    (X["Balance"] == 0) & (X["CreditScore"] < threshold)
                ).astype(int)
                for percentile, threshold in zip(
                    self.percentiles, self.credit_score_percentile
                )
            }
        )

        new_features.update(
            {
                f"IsMultiProductHighBalance_{num_products}": (
                    (X["NumOfProducts"] > num_products) & (X["Balance"] > 50_000)
                ).astype(int)
                for num_products in self.num_products
            }
        )

        new_features.update(
            {
                f"IsHighBalance_{percentile}": (X["Balance"] > threshold).astype(int)
                for percentile, threshold in zip(
                    self.percentiles, self.balance_percentile
                )
            }
        )

        new_features.update(
            {
                f"IsYounghighCreditScore_{age}y": (X["Age"] < age).astype(int)
                for age in self.ages
            }
        )
        new_features.update(
            {
                f"Balance_x_Germany": X["Balance"]
                * (X["Geography"] == "Germany").astype(int)
            }
        )
        new_features.update({f"HasBalance": (X["Balance"] > 0).astype(int)})

        # Retourner un DataFrame avec les nouvelles colonnes
        return pd.DataFrame(new_features, index=X.index)
