import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder

set_config(transform_output="pandas")


class ChurnRatios(BaseEstimator, TransformerMixin):

    def __init__(self, scale: bool = True):
        self.scaler = StandardScaler()
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        new_features = {}
        new_features["Tenure_Age_Ratio"] = X_copy["Tenure"] / (X_copy["Age"] + 1)
        new_features["Balance_Salary_Ratio"] = X_copy["Balance"] / (
            X_copy["EstimatedSalary"] + 1
        )
        df = pd.DataFrame(new_features, index=X_copy.index)
        if self.scale:
            df = self.scaler.fit_transform(df)
        return df


class ChurnProduct(BaseEstimator, TransformerMixin):

    def __init__(self, scale: bool = True):
        self.scaler = StandardScaler()
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        new_features = {}
        new_features["CreditScore_Age"] = X_copy["CreditScore"] * X_copy["Age"]
        new_features["Balance_NumProd"] = X_copy["Balance"] * X_copy["NumOfProducts"]
        new_features["Active_Balance"] = X_copy["Balance"] * X_copy["IsActiveMember"]
        new_features["Age_sq"] = X_copy["Age"] ** 2
        new_features["Age_cube"] = X_copy["Age"] ** 3
        df = pd.DataFrame(new_features, index=X_copy.index)
        if self.scale:
            df = self.scaler.fit_transform(df)
        return df


class ChurnCategories(BaseEstimator, TransformerMixin):

    def __init__(self, encode: bool = True):
        self.encode = encode

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        new_features = {}
        new_features["Age_Bin"] = pd.cut(
            X_copy["Age"],
            [17, 30, 40, 50, 60, 100],
            labels=["<30", "30-40", "40-50", "50-60", "60+"],
        )
        new_features["Tenure_Bin"] = pd.cut(
            X_copy["Tenure"],
            [0, 2, 3, 6, 100],
            labels=["<2", "2", "3-5", "6+"],
        )
        new_features["NumOfProducts_Bin"] = pd.cut(
            X_copy["NumOfProducts"],
            [0, 2, 3, 100],
            labels=["<2", "2", "3+"],
        )

        new_features["CreditScore_Bin"] = pd.cut(
            X_copy["CreditScore"],
            [350, 570, 650, 750, 900],
            labels=["≤570", "570-650", "650-750", "750+"],
        )

        new_features["Is_German"] = X_copy["Geography"] == "Germany"
        new_features["Gender"] = X_copy["Gender"]
        new_features["IsActiveMember"] = X_copy["IsActiveMember"]

        df = pd.DataFrame(new_features, index=X_copy.index)

        if self.encode:
            enc = ColumnTransformer(
                transformers=[
                    (
                        "onehot",
                        OneHotEncoder(drop="first", sparse_output=False),
                        [
                            "Is_German",
                            "Gender",
                            "IsActiveMember",
                            "Age_Bin",
                            "Tenure_Bin",
                            "NumOfProducts_Bin",
                        ],
                    )
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            )

            df = enc.fit_transform(df)

        return df
