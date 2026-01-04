import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

class QuantileCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.quantiles_ = {
            col: (X[col].quantile(self.lower), X[col].quantile(self.upper))
            for col in X.columns
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, (q_low, q_high) in self.quantiles_.items():
            X[col] = X[col].clip(q_low, q_high)
        return X

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.global_mean_ = y.mean()
        self.mapping_ = (
            X.assign(target=y)
            .groupby(self.col)['target']
            .mean()
        )
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.col] = X[self.col].map(self.mapping_)
        X[self.col] = X[self.col].fillna(self.global_mean_)
        return X
    


def preprocess_data(data, target_column, save_dir="bandung-house-price-preprocessing", save_data=True):
    os.makedirs(save_dir, exist_ok=True)
    data = data.copy()

    # =========================
    # 1. Drop kolom tidak relevan
    # =========================
    data = data.drop(columns=['house_name', 'building_area (m2)'], errors='ignore')

    # =========================
    # 2. Split fitur & target
    # =========================
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # 3. Log transform target
    # =========================
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    # =========================
    # 4. Numerical pipeline
    # =========================
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ('capper', QuantileCapper()),
        ('scaler', StandardScaler())
    ])

    X_train[numeric_features] = numeric_pipeline.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = numeric_pipeline.transform(X_test[numeric_features])

    # =====================
    # 5. Target Encoding
    # =====================
    te = TargetEncoder(col='location')
    X_train[['location']] = te.fit_transform(X_train[['location']], y_train)
    X_test[['location']] = te.transform(X_test[['location']])

      # =====================
    # 6. Simpan dataset hasil preprocessing
    # =====================
    if save_data:
        X_train.to_csv(f"{save_dir}/X_train_processed.csv", index=False)
        X_test.to_csv(f"{save_dir}/X_test_processed.csv", index=False)

        y_train.to_csv(f"{save_dir}/y_train_processed.csv", index=False)
        y_test.to_csv(f"{save_dir}/y_test_processed.csv", index=False)

    # =========================
    # 6. Simpan preprocessing
    # =========================
    dump(
        {
            "numeric_pipeline": numeric_pipeline,
            "target_encoder": te
        },
        f"preprocessor.joblib"
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = pd.read_csv("bandung-house-price.csv")

    X_train, X_test, y_train, y_test = preprocess_data(
        data=data,
        target_column='price',
        save_dir='preprocessing/bandung-house-price-preprocessing',
        save_data=True
    )
  