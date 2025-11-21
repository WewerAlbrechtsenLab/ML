import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils._metadata_requests import MetadataRequest

import pimmslearn.sampling
from pimmslearn.sklearn.ae_transformer import AETransformer
from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer
from inmoose.pycombat import pycombat_norm
from pathlib import Path

# class FeatureLocker:
#     """Locks feature names per CV fold"""
#     def _lock_features(self, X):
#         # Called during fit()
#         X_df = pd.DataFrame(X)
#         self.feature_names_ = list(X_df.columns)

#     def _enforce_features(self, X):
#         # Called during transform() – ensures consistent shape
#         X_df = pd.DataFrame(X)
#         return X_df.reindex(columns=self.feature_names_).values
    
class CFImputer(BaseEstimator, TransformerMixin):
    """
    Clean sklearn wrapper around pimmslearn CollaborativeFilteringTransformer.
    Works on wide-format proteomics matrices: rows = samples, cols = proteins.
    """

    def __init__(self,
                 n_factors=15,
                 epochs=20,
                 patience=3,
                 random_state=0):
        self.n_factors = n_factors
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

    # ----------------------- FIT -----------------------
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        X_df.index = X_df.index.get_level_values("sample_name")

        self.sample_index_ = X_df.index
        self.protein_cols_ = X_df.columns

        # Convert to long format
        long = (
            X_df.stack()
                .rename("intensity")
                .reset_index()
                .rename(columns={"sample_name": "sample",
                                 "level_1": "protein"})
                .set_index(["sample", "protein"])["intensity"]
        )

        # Sample train/val split for CF
        train_s, val_s = pimmslearn.sampling.sample_data(
            long,
            sample_index_to_drop=0,
            frac=0.9,
            random_state=self.random_state,
        )

        # Build and train CF model
        self.cf = CollaborativeFilteringTransformer(
            target_column="intensity",
            sample_column="sample",
            item_column="protein",
            n_factors=self.n_factors,
            out_folder="runs/cf",
            batch_size=4096,
        )

        self.cf.fit(
            train_s,
            val_s,
            epochs_max=self.epochs,
            cuda=False,
            patience=self.patience,
        )

        return self

    # ----------------------- TRANSFORM -----------------------
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df.index = X_df.index.get_level_values("sample_name")
        X_df = X_df.reindex(columns=self.protein_cols_)  # enforce consistent proteins

        # Convert to long format
        long = (
            X_df.stack()
                .rename("intensity")
                .reset_index()
                .rename(columns={"sample_name": "sample",
                                 "level_1": "protein"})
                .set_index(["sample", "protein"])["intensity"]
        )

        # Predict missing intensities
        long_imputed = self.cf.transform(long)

        # Reshape back to wide
        wide = long_imputed.unstack()
        wide = wide.reindex(index=self.sample_index_,
                            columns=self.protein_cols_)

        return wide.values

class VAEImputer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 hidden_layers=[512],
                 latent_dim=30,
                 epochs=50,
                 patience=5,
                 random_state=0):
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        X_df = pd.DataFrame(X).copy()
        self.feature_names_ = list(X_df.columns)

        #self._lock_features(X_df)  # lock columns from TRAIN fold only

        self.vae = AETransformer(
            hidden_layers=self.hidden_layers,
            latent_dim=self.latent_dim,
            model="VAE",
            out_folder="runs/vae",
            batch_size=64,
        )

        # train VAE on TRAIN-FOLD only
        self.vae.fit(
            X_df,
            y=None,
            epochs_max=self.epochs,
            cuda=False,
            patience=None,
        )

        return self

    def transform(self, X, **kwargs):
        X_df = pd.DataFrame(X).copy()

        # enforce same column universe
        X_df = pd.DataFrame(
            X_df, columns=self.feature_names_
        )

        # run VAE imputation
        out = self.vae.transform(X_df)

        # enforce feature order again
        out = pd.DataFrame(out, columns=self.feature_names_)

        return out.values
    
    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y=y, **kwargs)
        return self.transform(X, **kwargs)

class CombatCorrector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.batch_labels_ = None
        self.batch_params_ = None
        self.feature_names_ = None
        self._skip_ = False
        self.set_fit_request(batch_labels=True)
        self.set_transform_request(batch_labels=True)

    def fit(self, X, y=None, *, batch_labels=None):
        if batch_labels is None:
            raise ValueError("batch_labels must be passed to CombatCorrector.fit")

        self.batch_labels_ = np.asarray(batch_labels)
        X_df = pd.DataFrame(X).copy()
        self.feature_names_ = list(X_df.columns)

        # check for invalid batches
        unique, counts = np.unique(self.batch_labels_, return_counts=True)
        if (counts < 2).any():
            # mark as "do nothing"
            self._skip_ = True
            return self

        # genes x samples for pycombat_norm
        X_df_t = X_df.T

        # estimate params once on training data
        self.batch_params_ = pycombat_norm(
            X_df_t,
            batch=self.batch_labels_,
            return_params=True,
        )
        self._skip_ = False
        return self

    def transform(self, X, *, batch_labels=None):
        if self._skip_ or self.batch_params_ is None:
            return np.asarray(X)

        if batch_labels is None:
            raise ValueError("batch_labels must be passed to CombatCorrector.transform")

        batch_labels = np.asarray(batch_labels)

        X_df = pd.DataFrame(X).copy()
        # make sure same feature order as in fit
        if self.feature_names_ is not None:
            X_df = pd.DataFrame(X_df, columns=self.feature_names_)
        X_df_t = X_df.T

        corrected = pycombat_norm(
            X_df_t,
            batch=batch_labels,
            params=self.batch_params_,
        )

        out = corrected.T
        if self.feature_names_ is not None:
            out = out.reindex(columns=self.feature_names_)
        return out.to_numpy()
    
    def fit_transform(self, X, y=None, *, batch_labels=None, **fit_params):
        # IMPORTANT: forward batch_labels to BOTH fit and transform
        self.fit(X, y=y, batch_labels=batch_labels, **fit_params)
        return self.transform(X, batch_labels=batch_labels)

def build_fold_preprocessor(batch_labels):
    combat = CombatCorrector()

    # === Tell CombatCorrector that it expects metadata ===
    combat.set_fit_request(batch_labels=True)
    combat.set_transform_request(batch_labels=True)

    # === Let Pipeline route metadata to the combat step ===
    pre = Pipeline([
        ("vae", VAEImputer()),
        ("combat", combat),
    ])

    return pre
