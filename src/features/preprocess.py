from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import zscore


def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


class RowwiseZScoreScaler(BaseEstimator, TransformerMixin):
    """Scales each sample (row) to mean 0 and std 1 across its features."""
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return zscore(X, axis=1, nan_policy='omit')

    def get_feature_names_out(self, input_features=None):
        # This transformer preserves the column layout, so propagate the incoming names.
        return input_features


def build_preprocessor(features: pd.DataFrame) -> Pipeline:
    
    numeric_cols, categorical_cols = infer_column_types(features)
    
    numeric_pipeline = Pipeline(
        steps=[
            #("impute", SimpleImputer(strategy="median")),
            ("row_zscore", RowwiseZScoreScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            #("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    processor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )

    return processor
