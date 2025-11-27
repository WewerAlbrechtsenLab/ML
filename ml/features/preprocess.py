import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as SklearnPipeline

from MSprocessing.preprocessing.normalization import normalize_sample
from pimmslearn.sklearn.ae_transformer import AETransformer
from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer
from inmoose.pycombat import pycombat_norm
from ml.utils.__init__ import set_global_seed
from pimmslearn.sampling import sample_data
from pimmslearn.sklearn.cf_transformer import (
    TabularCollab, Categorify, TransformBlock,
    EmbeddingDotBias, MSELossFlat, IndexSplitter
)
from fastai.data.all import *
from fastai.learner import Learner
from fastai.callback.tracker import EarlyStoppingCallback
from ml.utils.config import PipelineConfig

class CFImputer(BaseEstimator, TransformerMixin):
    """
    sklearn wrapper around pimmslearn CollaborativeFilteringTransformer.
    Works on wide-format proteomics matrices: rows = samples, cols = proteins.
    """

    def __init__(self,
                 n_factors=15,
                 epochs=20,
                 patience=3,
                 random_state=43,):
        self.n_factors = n_factors
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

    # ----------------------- FIT -----------------------
    def fit(self, X, y, **kwargs):
        set_global_seed(self.random_state)
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X_df.index = X_df.index.get_level_values("sample")

        self.sample_index_ = X_df.index
        self.feature_names_ = X_df.columns

        # Convert to long format
        long = (
            X_df
            .stack(dropna=True)
            .reset_index()          
            .rename(columns={"level_0": "sample",
                            "level_1": "protein",
                            0: "intensity"})
        )

        # Ensure strings (fastai requires categorical string IDs)
        long["sample"] = long["sample"].astype(str)
        long["protein"] = long["protein"].astype(str)
        long = long.set_index(["sample", "protein"])["intensity"]
        long = long.to_frame(name="intensity")

        # Sample train/val split for CF
        train_s, val_s = sample_data(
            long["intensity"],
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
        # Train using monkey-patched fit()
        self.cf.fit(
            train_s,
            y = val_s, #No val
            epochs_max=self.epochs,
            cuda=False,
            patience=None,
        )

        return self

    # ----------------------- TRANSFORM -----------------------
    def transform(self, X, **kwargs):
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X_df.index = X_df.index.get_level_values("sample")
        X_df = X_df.reindex(columns=self.feature_names_) 

        # Convert to long format
        long = (
            X_df
            .stack(dropna=False)
            .reset_index()          
            .rename(columns={"level_0": "sample",
                     "level_1": "protein",
                     0: "intensity"})
)

        # Ensure strings (fastai requires categorical string IDs)
        long["sample"] = long["sample"].astype(str)
        long["protein"] = long["protein"].astype(str)
        long = long.set_index(["sample", "protein"])["intensity"]

        # Predict missing intensities
        long_imputed = self.cf.transform(long)
        # Collapse duplicates 
        long_imputed = long_imputed.groupby(level=[0,1]).max()

        # Reshape back to wide
        wide = long_imputed.unstack()

        return wide.values
    
    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y=y, **kwargs).transform(X, **kwargs)

class VAEImputer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 hidden_layers=[512],
                 latent_dim=30,
                 epochs=50,
                 patience=5,
                 random_state=43,
                 frac=0.1):
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.frac = frac
        

    def fit(self, X, y=None, **kwargs):
        set_global_seed(self.random_state)

        X_df = pd.DataFrame(X).copy()
        index_cols = list(X_df.index.names)
        feature_names_ = list(X_df.columns)
        df_work = (
                X_df.reset_index()
                .drop([c for c in index_cols if c != "sample"], axis=1)
                .set_index("sample")
            )
        df_work.index.name = "sample"
        df_work.columns.name = "protein"


            
        val_X, train_X = sample_data(df_work.stack(),
                                    sample_index_to_drop=0,
                                    weights=X_df.notna().sum(),)
        val_X, train_X = val_X.unstack(), train_X.unstack()
        val_X = pd.DataFrame(pd.NA, index=train_X.index, columns=train_X.columns).fillna(val_X)
 

        # Create AE model AFTER seeding
        self.vae = AETransformer(
            hidden_layers=self.hidden_layers,
            latent_dim=self.latent_dim,
            model="VAE",
            out_folder="runs/vae",
            batch_size=64,
        )

        # Fit VAE 
        self.vae.fit(
            train_X,
            y=val_X,
            epochs_max=self.epochs,
            cuda=False,
            patience=None,
        )

        return self

    def transform(self, X, **kwargs):
        X_df = pd.DataFrame(X).copy()
        index_cols = list(X_df.index.names)
        
        df_work = (
                X_df.reset_index()
                .drop([c for c in index_cols if c != "sample"], axis=1)
                .set_index("sample")
            )
        df_work.index.name = "sample"
        df_work.columns.name = "protein"

        # run VAE imputation
        out = self.vae.transform(df_work)
        out = pd.DataFrame(out,
                             index=df_work.index,
                             columns=df_work.columns)

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

class Normalizer(BaseEstimator,TransformerMixin):
    def __init__(self, method: str = "cscore", round_digits: int = 3):
        self.method = method
        self.round_digits = round_digits
        self.feature_names_ = None

    def fit(self, X, y=None, **kwargs): 
        X_df = pd.DataFrame(X).copy()
        self.feature_names_ = list(X_df.columns)
        return self
    def transform(self, X, **kwargs):
        X_df = pd.DataFrame(X, columns=self.feature_names_)

        X_norm = normalize_sample(
            X_df,
            method=self.method,
            round_digits=self.round_digits,
        )

        return X_norm.to_numpy()

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y=y, **kwargs).transform(X, **kwargs)
    

def build_fold_preprocessor(batch_labels):
    combat = CombatCorrector()

    # === Tell CombatCorrector that it expects metadata ===
    combat.set_fit_request(batch_labels=True)
    combat.set_transform_request(batch_labels=True)

    pre = SklearnPipeline([
        #("cf", CFImputer()),
        ("vae", VAEImputer()),  
        ("combat", combat),
        ("norm", Normalizer()),
    ])

    return pre
