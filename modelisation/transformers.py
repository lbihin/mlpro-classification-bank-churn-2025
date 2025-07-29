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
        self.weights_by_country_ = {
            "France": [0.50512678, -0.09958729, -0.62135048, 0.40087766, 0.2104039],
            "Spain": [0.53319479, -0.09586698, -0.56027375, 0.4197214, 0.1958134],
            "Germany": [0.408, -0.57, -0.51, 0.306, 0.204],
        }
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
                    print(
                        f"[WARN] Aucun échantillon pour {country}, poids par défaut appliqués."
                    )
                self.weights_by_country_[country] = [1, 1, 1, 1, 1]
                self.threshold_by_country_[country] = self.threshold
                continue

            # init_params = [0.4, -0.6, -0.5, 0.3, 0.2, self.threshold]
            init_params = [*self.weights_by_country_.get(country, []), self.threshold]

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
                    return self._objective(
                        np.append(w, self.threshold), X_country, y_country, country
                    )

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
        unknown_countries = set(X_out["Geography"].unique()) - set(
            self.weights_by_country_.keys()
        )
        if unknown_countries:
            raise ValueError(
                f"Pays inconnus détectés pendant transform : {unknown_countries}"
            )

        churn_scores = np.zeros(len(X_out))

        for country, weights in self.weights_by_country_.items():
            mask = X_out["Geography"] == country
            churn_scores[mask] = self._compute_score(X_out[mask], weights, country)

        return pd.DataFrame({"churn_score": churn_scores}, index=X_out.index)


class ChurnFeature(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        # bins = [0, 35, 50, 100]
        # labels = ["Jeune", "Adulte", "Senior"]

        new_features = {
            # "NumOfProducts_by_Age": X_copy.NumOfProducts / (1 + X_copy.Age),
            "Balance_by_NumOfProducts": X_copy.Balance / (1 + X_copy.NumOfProducts),
            "NumOfProducts^2": X_copy.NumOfProducts**2,
            "Age_x_IsActiveMember": X_copy.Age / (1 + X_copy.IsActiveMember),
            "Is_Germany": X_copy.Geography == "Germany",
            "Has_Balance": X_copy.Balance == 0,
            "Germany_Inactive_HighBalance": (X_copy.Geography == "Germany")
            & (X_copy.IsActiveMember == 0)
            & (X_copy.Balance > 0).astype(int),
            # "AgeGroup": pd.cut(
            #     X_copy["Age"], bins=bins, labels=labels, right=False
            # ).astype(str),
        }

        return pd.DataFrame(new_features, index=X_copy.index)
