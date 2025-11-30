import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

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
        # new_features["Tenure_Age_Ratio"] = X_copy["Tenure"] / (X_copy["Age"] + 1)
        # new_features["Tenure_Age_Ratio"] = X_copy["Tenure"] / (X_copy["Age"] + 1)
        new_features["Balance_Salary_Ratio"] = X_copy["EstimatedSalary"] / (
            X_copy["Balance"] + 1
        )
        new_features["Balance_Age_Ratio"] = X_copy["Age"] * (X_copy["Balance"] + 1)

        df = pd.DataFrame(new_features, index=X_copy.index)
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
        # new_features["Age_Bin"] = pd.cut(
        #     X_copy["Age"],
        #     [17, 30, 40, 50, 60, 100],
        #     labels=["Age_<30", "Age_30-40", "Age_40-50", "Age_50-60", "Age_60+"],
        # )
        new_features["Tenure_Bin"] = pd.cut(
            X_copy["Tenure"],
            [0, 2, 3, 6, 100],
            labels=["Tenure_≤1", "Tenure_2", "Tenure_3-5", "Tenure_6+"],
            include_lowest=True,
        )
        new_features["NumOfProducts_Bin"] = pd.cut(
            X_copy["NumOfProducts"],
            [0, 2, 3, 100],
            labels=["NumOfProducts_≤1", "NumOfProducts_2", "NumOfProducts_3+"],
        )

        new_features["CreditScore_Bin"] = pd.cut(
            X_copy["CreditScore"],
            [350, 570, 650, 750, 900],
            labels=["CreditScore_≤570", "CreditScore_570-650", "CreditScore_650-750", "CreditScore_750+"],
        )

        new_features["Is_German"] = X_copy["Geography"] == "Germany"
        new_features["Gender"] = X_copy["Gender"]
        # new_features["IsActiveMember"] = X_copy["IsActiveMember"]

        df = pd.DataFrame(new_features, index=X_copy.index)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [
                "Is_German",
                "Gender_Male",
                # "Age_Bin",
                "Tenure_Bin",
                "NumOfProducts_Bin",
                "CreditScore_Bin",
            ]
        )
