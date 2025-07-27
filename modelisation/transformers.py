import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score


class ChurnProbabilityScore(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        threshold=1.0,
        countries="all",
        optimize_threshold=True,
        optimizer="Nelder-Mead",
        verbose=False,
    ):
        self.threshold = threshold
        self.countries = countries
        self.optimize_threshold = optimize_threshold
        self.optimizer = optimizer
        self.verbose = verbose
        self.weights_by_country_ = {}
        self.threshold_by_country_ = {}
        self.max_age_ = None  # pour normalisation

    def _compute_score(self, X, weights, country):
        w_age, w_num_products, w_active, w_has_balance, w_geography = weights

        has_balance = (X["Balance"] > 0).astype(int)
        geography_match = (X["Geography"].astype(str) == country).astype(int)
        age_norm = (X["Age"] - 18) / (self.max_age_ - 18)

        score = (
            w_age * age_norm
            + w_num_products * X["NumOfProducts"]
            + w_active * X["IsActiveMember"]
            + w_has_balance * has_balance
            + w_geography * geography_match
        )

        return 1 / (1 + np.exp(-score))

    def _objective(self, params, X, y, country):
        weights = params[:-1]
        threshold = np.clip(params[-1], 0.1, 0.9)  # contrainte

        prob = self._compute_score(X, weights, country)
        preds = (prob >= threshold).astype(int)
        return -f1_score(y, preds)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        X["Geography"] = X["Geography"].astype(str)

        self.max_age_ = X["Age"].max()

        countries_to_fit = (
            X["Geography"].unique() if self.countries == "all" else self.countries
        )

        for country in countries_to_fit:
            mask = X["Geography"] == country
            X_country, y_country = X[mask], y[mask]

            if len(X_country) == 0:
                if self.verbose:
                    print(f"[WARN] Aucun échantillon pour {country}, poids par défaut appliqués.")
                self.weights_by_country_[country] = [1, 1, 1, 1, 1]
                self.threshold_by_country_[country] = self.threshold
                continue

            init_params = [0.4, -0.6, -0.5, 0.3, 0.2, self.threshold]

            if self.optimize_threshold:
                res = minimize(
                    self._objective,
                    init_params,
                    args=(X_country, y_country, country),
                    method=self.optimizer,
                    options={"maxiter": 300},
                )
                weights = res.x[:-1]
                threshold = np.clip(res.x[-1], 0.1, 0.9)
            else:
                # Seuil fixe : on n’optimise que les poids
                def fixed_threshold_objective(w):
                    return self._objective(np.append(w, self.threshold), X_country, y_country, country)

                res = minimize(
                    fixed_threshold_objective,
                    init_params[:-1],
                    method=self.optimizer,
                    options={"maxiter": 300},
                )
                weights = res.x
                threshold = self.threshold

            self.weights_by_country_[country] = weights
            self.threshold_by_country_[country] = threshold

            if self.verbose:
                print(f"[{country}] Weights={weights}, Threshold={threshold:.2f}")

        return self

    def transform(self, X: pd.DataFrame):
        if not self.weights_by_country_:
            raise ValueError("Le transformeur doit être fit() avant transform().")

        X_out = X.copy()
        X_out["Geography"] = X_out["Geography"].astype(str)

        # Vérifier pays non vus
        unknown_countries = set(X_out["Geography"].unique()) - set(self.weights_by_country_.keys())
        if unknown_countries:
            raise ValueError(f"Pays inconnus détectés pendant transform : {unknown_countries}")

        churn_scores = np.zeros(len(X_out))

        for country, weights in self.weights_by_country_.items():
            mask = X_out["Geography"] == country
            churn_scores[mask] = self._compute_score(X_out[mask], weights, country)

        return pd.DataFrame({"churn_score": churn_scores}, index=X_out.index)


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
