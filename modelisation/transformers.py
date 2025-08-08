from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
from numpy import ndarray
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns, method="iqr", factor=1.5):
        self.columns = columns
        self.method = method
        self.factor = factor

    def fit(self, X, y=None):
        # On stocke les bornes par colonne
        self.bounds_ = {}
        for col in self.columns:
            data = X[col]

            if self.method == "iqr":
                Q1 = np.nanpercentile(data, 25)
                Q3 = np.nanpercentile(data, 75)
                IQR = Q3 - Q1
                lower = Q1 - self.factor * IQR
                upper = Q3 + self.factor * IQR
            elif self.method == "zscore":
                mean = data.mean()
                std = data.std()
                lower = mean - self.factor * std
                upper = mean + self.factor * std
            else:
                raise ValueError("Méthode inconnue : choisir 'iqr' ou 'zscore'")

            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X, y=None):
        # Création du masque
        mask = pd.Series(True, index=X.index)
        for col, (lower, upper) in self.bounds_.items():
            mask &= X[col].between(lower, upper)

        X_filtered = X[mask]
        y_filtered = y[mask] if y is not None else None

        # Si utilisé dans imblearn.Pipeline → retourne (X, y)
        return (X_filtered, y_filtered) if y is not None else X_filtered


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


class ChurnFeature(TransformerMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...

    def fit(self, X, y=None):
        X_copy = X.copy()

        self.grp = X_copy.groupby("Geography")["Balance"].transform("mean")
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        # bins = [0, 35, 50, 100]
        # labels = ["Jeune", "Adulte", "Senior"]

        new_features = {
            "Is_Germany": X_copy.Geography
            == "Germany",  # Le ratio churn est différent entre
            # "NumOfProducts_by_Age": X_copy.NumOfProducts / (1 + X_copy.Age),
            "Balance_by_NumOfProducts": X_copy.Balance / (1 + X_copy.NumOfProducts),
            "NumOfProducts^2": X_copy.NumOfProducts**2,
            "Age_x_IsActiveMember": X_copy.Age / (1 + X_copy.IsActiveMember),
            "Age_x_Balance": X_copy.Age * X_copy.Balance,
            "Germany_Inactive_HighBalance": (X_copy.Geography == "Germany")
            & (X_copy.IsActiveMember == 0)
            & (X_copy.Balance > 0).astype(int),
            "Balance_diff_Geography": X_copy.Balance - self.grp,
            "Balance_ratio_Geography": X_copy.Balance / (1 + self.grp),
            "Balance_by_EstimatedSalary": X_copy.Balance / (1 + X_copy.EstimatedSalary),
            "EstimatedSalary_by_Age": X_copy.EstimatedSalary / (1 + X_copy.Age),
            "Age_by_Tenure": X_copy.Age / (1 + X_copy.Tenure),
            "NumOfProducts_by_Tenure": X_copy.NumOfProducts / (1 + X_copy.Tenure),
        }

        return pd.DataFrame(new_features, index=X_copy.index)


class SplineTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer scikit-learn pour générer des bases B-splines d'une variable continue.
    Renvoie uniquement les colonnes spline générées.

    Paramètres :
      - feature_name (str) : nom de la colonne à transformer
      - knots (tuple) : positions des nœuds internes
      - degree (int) : degré du polynôme (généralement 3 pour cubique)
      - include_intercept (bool) : inclure l'intercept dans la matrice de design
      - prefix (str) : préfixe pour les noms de colonnes générées
    """

    def __init__(
        self,
        feature_name="Age",
        knots=(30, 50, 70),
        degree=3,
        include_intercept=False,
        prefix="spline",
    ):
        self.feature_name = feature_name
        self.knots = knots
        self.degree = degree
        self.include_intercept = include_intercept
        self.prefix = prefix

    def fit(self, X, y=None):
        # Pas d'apprentissage nécessaire
        return self

    def transform(self, X) -> pd.DataFrame:
        # Attend un DataFrame pandas contenant feature_name
        if self.feature_name not in X.columns:
            raise ValueError(f"Colonne '{self.feature_name}' introuvable dans X")

        # Construction de la matrice de design B-spline
        spl = patsy.dmatrix(
            f"bs({self.feature_name}, knots={self.knots}, degree={self.degree}, include_intercept={self.include_intercept})",
            {self.feature_name: X[self.feature_name]},
            return_type="dataframe",
        )
        # Renommer les colonnes pour éviter les conflits
        spl.columns = [
            f"{self.prefix}_{i}_{self.feature_name}" for i in range(spl.shape[1])
        ]
        spl.index = X.index

        # Retourner uniquement les colonnes spline générées
        return spl


class LdaScore(BaseEstimator, TransformerMixin):
    """Produit 1 colonne : le score LDA (decision_function)."""

    def __init__(self, num_cols, cat_cols, n_components=1):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.n_components = n_components

    def _make_prep(self):
        return ColumnTransformer(
            [
                ("num", StandardScaler(), self.num_cols),
                ("cat", OrdinalEncoder(), self.cat_cols),
            ],
            remainder="drop",
        )

    # ------------------------------------------------------------------
    def fit(self, X, y):
        self.prep_ = self._make_prep()
        Xprep = self.prep_.fit_transform(X, y)
        self.lda_ = LinearDiscriminantAnalysis(n_components=self.n_components)
        self.lda_.fit(Xprep, y)
        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        Xprep = self.prep_.transform(X)
        scores = self.lda_.decision_function(Xprep).reshape(-1, 1)
        return scores  # ndarray (n_samples, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["score_lda"])


class SupervisedBinner(BaseEstimator, TransformerMixin):
    """
    Coupe chaque variable numérique en intervalles optimisés pour la cible
    (DecisionTreeClassifier 1-D → id de feuille).
    """

    def __init__(self, max_depth=3, min_samples_leaf=200, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        X = pd.DataFrame(X)  # garde les noms si dispo
        self.features_ = X.columns.tolist()
        self.trees_ = {}
        self.bucket_maps_ = {}

        for col in self.features_:
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            ).fit(X[[col]], y)
            self.trees_[col] = tree

            # id de feuille → rang 0…n-1  (plus lisible)
            leaves = tree.apply(X[[col]])
            uniq = np.unique(leaves)
            self.bucket_maps_[col] = {leaf: i for i, leaf in enumerate(uniq)}

        return self

    # ------------------------------------------------------------------ #
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.features_)
        out = {}
        for col in self.features_:
            leaf = self.trees_[col].apply(X[[col]])
            out[f"{col}_bin_sup"] = np.vectorize(self.bucket_maps_[col].get)(leaf)
        return pd.DataFrame(out).to_numpy()

    # ------------------------------------------------------------------ #
    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_bin_sup" for c in self.features_])
