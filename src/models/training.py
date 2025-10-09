from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import RFE, RFECV, SelectKBest, mutual_info_classif
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.models.metrics import scoring_map
from src.utils.config import PipelineConfig
from src.utils.run_logger import log_training_run
from functools import partial

def build_outer_cv(config: PipelineConfig) -> StratifiedKFold:
    outer = getattr(config, "outer_splits", None)
    if outer is None:
        outer = getattr(config, "n_splits", 5)
    return StratifiedKFold(
        n_splits=outer,
        shuffle=True,
        random_state=config.random_state,
    )


def build_inner_cv(config: PipelineConfig) -> StratifiedKFold:
    inner = getattr(config, "inner_splits", None)
    if inner is None:
        base = getattr(config, "n_splits", getattr(config, "outer_splits", 5))
        inner = max(2, int(base) // 2) if base else 2
    return StratifiedKFold(
        n_splits=inner,
        shuffle=True,
        random_state=config.random_state,
    )


def _ensure_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    return pd.DataFrame(X)


def _ensure_series(y) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Nested CV currently supports a single target column.")
        return y.iloc[:, 0].reset_index(drop=True)
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True)
    return pd.Series(y)
def _default_rfe_values(n_features: int) -> List[int]:
    if n_features <= 1:
        return [1]
    fractions = [0.25, 0.5, 0.75, 1.0]
    values = [max(1, int(round(n_features * frac))) for frac in fractions]
    unique: List[int] = []
    for value in values:
        value = min(n_features, value)
        if value not in unique:
            unique.append(value)
    if n_features not in unique:
        unique.append(n_features)
    return unique


def _sanitize_param_grid(
    grid: Dict[str, Any], n_features: int, include_rfe: bool
) -> Dict[str, List[Any]]:
    sanitized: Dict[str, List[Any]] = {}
    for key, raw_values in grid.items():
        if isinstance(raw_values, (list, tuple)):
            values = list(raw_values)
        else:
            values = [raw_values]

        if key == "select__k":
            raise ValueError(
                "Manual configuration of select__k is no longer supported. "
                "Set feature_selection='univariate' to enable automatic selection."
            )
        elif key == "rfe__n_features_to_select":
            filtered_rfe: List[int] = []
            for value in values:
                if value is None:
                    filtered_rfe.append(n_features)
                elif isinstance(value, str) and value.lower() == "all":
                    filtered_rfe.append(n_features)
                elif isinstance(value, int) and 1 <= value <= n_features:
                    filtered_rfe.append(value)
            if not filtered_rfe:
                filtered_rfe = _default_rfe_values(n_features)
            sanitized[key] = filtered_rfe
        else:
            sanitized[key] = values

    if include_rfe and "rfe__n_features_to_select" not in sanitized:
        sanitized["rfe__n_features_to_select"] = _default_rfe_values(n_features)
    return sanitized


def _resolve_search_space(
    model_name: str, config: PipelineConfig, n_features: int
) -> Dict[str, List[Any]]:
    user_spaces = getattr(config, "search_spaces", {}) or {}
    include_rfe = getattr(config, "use_rfe", False)
    if model_name in user_spaces:
        return _sanitize_param_grid(
            user_spaces[model_name], n_features, include_rfe
        )

    defaults: Dict[str, Any] = {}
    if model_name == "logistic_regression":
        defaults = {
            "estimator__C": [0.01, 0.1, 1.0, 10.0],
            "estimator__solver": ["lbfgs", "saga"],
        }
    elif model_name == "random_forest":
        defaults = {
            "estimator__n_estimators": [200, 400, 600],
            "estimator__max_depth": [None, 10, 20],
            "estimator__min_samples_split": [2, 5],
        }
    return _sanitize_param_grid(defaults, n_features, include_rfe)


def _selected_feature_count(feature_selection: str, pipeline: Pipeline) -> int | None:
    """Return the number of features retained by the fitted feature selector."""
    if feature_selection == "univariate":
        selector = pipeline.named_steps.get("select")
        if selector is None:
            return None
        if hasattr(selector, "get_support"):
            support = selector.get_support()
            if support is not None:
                return int(np.sum(support))
        k = getattr(selector, "k", None)
        if isinstance(k, int):
            return k
        return None
    if feature_selection == "rfe":
        rfe = pipeline.named_steps.get("rfe")
        if rfe is None:
            return None
        support = getattr(rfe, "support_", None)
        if support is not None:
            return int(np.sum(support))
        n_features = getattr(rfe, "n_features_", None)
        if isinstance(n_features, int):
            return n_features
        n_select = getattr(rfe, "n_features_to_select", None)
        if isinstance(n_select, int):
            return n_select
        return None
    if feature_selection == "rfecv":
        selector = pipeline.named_steps.get("feature_select") or pipeline.named_steps.get("rfecv")
        if selector is None:
            return None
        support = getattr(selector, "support_", None)
        if support is None and hasattr(selector, "get_support"):
            support = selector.get_support()
        if support is not None:
            return int(np.sum(support))
        return None
    return None


def _feature_names_from_preprocessor(preprocessor, fallback) -> List[str]:
    if preprocessor is None:
        return list(fallback)
    get_names = getattr(preprocessor, "get_feature_names_out", None)
    names = None
    if callable(get_names):
        try:
            names = get_names()
        except TypeError:
            names = get_names(fallback)
        except Exception:
            names = None
    if names is None:
        return list(fallback)
    if isinstance(names, (list, tuple)):
        return list(names)
    if hasattr(names, "tolist"):
        return list(names.tolist())
    return list(names)


def _selected_feature_names(
    feature_selection: str, pipeline: Pipeline, input_columns: List[str]
) -> List[str] | None:
    if feature_selection not in {"univariate", "rfe", "rfecv"}:
        return None

    preprocessor = pipeline.named_steps.get("preprocess")
    feature_names = _feature_names_from_preprocessor(preprocessor, input_columns)

    if feature_selection == "univariate":
        selector = pipeline.named_steps.get("select")
        if selector is None:
            return None
        if hasattr(selector, "get_support"):
            support = selector.get_support()
            if support is not None:
                support = np.asarray(support, dtype=bool)
                return [name for name, keep in zip(feature_names, support) if keep]
        k = getattr(selector, "k", None)
        if isinstance(k, int):
            return feature_names[:k]
        return None

    if feature_selection == "rfe":
        rfe = pipeline.named_steps.get("rfe")
        if rfe is None:
            return None
        support = getattr(rfe, "support_", None)
        if support is not None:
            support = np.asarray(support, dtype=bool)
            return [name for name, keep in zip(feature_names, support) if keep]
        n_features = getattr(rfe, "n_features_", None)
        if isinstance(n_features, int):
            return feature_names[:n_features]
        n_select = getattr(rfe, "n_features_to_select", None)
        if isinstance(n_select, int):
            return feature_names[:n_select]
        return None

    if feature_selection == "rfecv":
        selector = pipeline.named_steps.get("feature_select") or pipeline.named_steps.get("rfecv")
        if selector is None:
            return None
        support = getattr(selector, "support_", None)
        if support is None and hasattr(selector, "get_support"):
            support = selector.get_support()
        if support is not None:
            support = np.asarray(support, dtype=bool)
            return [name for name, keep in zip(feature_names, support) if keep]
        return None

    return feature_names


class FixedFeatureSelector(BaseEstimator, TransformerMixin):
    """Apply a precomputed feature mask without refitting feature selection."""

    def __init__(self, support_mask):
        self.support_mask = np.asarray(support_mask, dtype=bool)

    def fit(self, X, y=None):
        if X.shape[1] != self.support_mask.size:
            raise ValueError("Support mask length does not match feature dimension.")
        return self

    def transform(self, X):
        return X[:, self.support_mask]

    def get_support(self):
        return self.support_mask

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return input_features
        return [name for name, keep in zip(input_features, self.support_mask) if keep]


def _build_pipeline(feature_selection: str, preprocessor, estimator, scoring: str | None = None, cv_splits: int = 5):
    steps = [("preprocess", clone(preprocessor))]
    if feature_selection == "univariate":
        steps.append(("select", SelectKBest(score_func=partial(mutual_info_classif, random_state=PipelineConfig.random_state))))
    elif feature_selection == "rfe":
        steps.append(("rfe", RFE(estimator=clone(estimator))))
    elif feature_selection == "rfecv":
        steps.append(("rfecv", RFECV(estimator=clone(estimator), cv=cv_splits, scoring=scoring, step=1, min_features_to_select=1)))
    # "none" just skips feature selection
    steps.append(("estimator", clone(estimator)))
    return Pipeline(steps=steps)



def _finalize_rfecv_pipeline(pipeline: Pipeline, X: pd.DataFrame, y, config: PipelineConfig, scoring_code: str | None):
    if "rfecv" not in pipeline.named_steps:
        return pipeline

    rfecv_step = pipeline.named_steps["rfecv"]
    support = getattr(rfecv_step, "support_", None)
    if support is None:
        return pipeline

    preprocessor = pipeline.named_steps.get("preprocess")
    estimator = pipeline.named_steps.get("estimator")
    fixed_selector = FixedFeatureSelector(support)

    finalized_steps = []
    if preprocessor is not None:
        finalized_steps.append(("preprocess", clone(preprocessor)))
    finalized_steps.append(("feature_select", fixed_selector))
    if estimator is not None:
        finalized_steps.append(("estimator", clone(estimator)))

    finalized = Pipeline(finalized_steps)
    finalized.fit(X, y)
    finalized.selected_support_ = support
    finalized.rfecv_cv_results_ = getattr(rfecv_step, "cv_results_", None)
    return finalized

def nested_cross_validate_models(
    models: Dict[str, BaseEstimator],
    preprocessor,
    X,
    y,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, Dict[str, BaseEstimator]]:
    scoring = scoring_map(config.task_type)
    if not scoring:
        raise ValueError(f"No scoring metrics defined for task type '{config.task_type}'.")
    primary_metric = next(iter(scoring))

    X_df = _ensure_frame(X)
    y_series = _ensure_series(y)

    if y_series.isna().any():
        valid_mask = ~y_series.isna()
        X_df = X_df.loc[valid_mask].reset_index(drop=True)
        y_series = y_series.loc[valid_mask].reset_index(drop=True)

    label_encoder: LabelEncoder | None = None
    if config.task_type == "binary":
        unique_labels = pd.Index(y_series.unique())
        if len(unique_labels) != 2:
            raise ValueError(
                "Binary classification requires exactly two classes in the target. "
                f"Observed {len(unique_labels)} unique labels."
            )
        needs_encoding = not set(unique_labels).issubset({0, 1})
        if needs_encoding:
            label_encoder = LabelEncoder()
            y_series = pd.Series(
                label_encoder.fit_transform(y_series), index=y_series.index
            )

    outer_cv = build_outer_cv(config)
    inner_cv = build_inner_cv(config)
    scorers = {name: get_scorer(code) for name, code in scoring.items()}

    records: List[Dict[str, Any]] = []
    fold_history: Dict[str, List[Dict[str, Any]]] = {}
    best_estimators: Dict[str, BaseEstimator] = {}

    feature_selection = getattr(config, "feature_selection", "none")

    for model_name, estimator in models.items():
        fold_scores: Dict[str, List[float]] = {metric: [] for metric in scoring}
        fold_history[model_name] = []
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_df, y_series)):
            X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
            y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]

            pipeline = _build_pipeline(
                feature_selection,
                preprocessor,
                estimator,
                scoring.get(primary_metric),
                inner_cv.get_n_splits(),
            )

            param_grid = _resolve_search_space(model_name, config, X_train.shape[1])

            if feature_selection != "rfe" and "rfe__n_features_to_select" in param_grid:
                param_grid = {
                    k: v for k, v in param_grid.items() if k != "rfe__n_features_to_select"
                }

            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring=scoring,
                refit=primary_metric,
                cv=inner_cv,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)

            best_pipeline = search.best_estimator_
            best_pipeline = _finalize_rfecv_pipeline(best_pipeline, X_train, y_train, config, scoring.get(primary_metric))
            fold_result: Dict[str, Any] = {
                "outer_fold": fold_idx,
                "best_params": search.best_params_,
            }

            feature_count = _selected_feature_count(feature_selection, best_pipeline)
            if feature_count is None:
                feature_count = X_train.shape[1]
            fold_result["selected_feature_count"] = feature_count

            selected_features = None
            if feature_selection in {"univariate", "rfe", "rfecv"}:
                selected_features = _selected_feature_names(
                    feature_selection,
                    best_pipeline,
                    list(X_train.columns),
                )
            fold_result["selected_features"] = selected_features

            for metric_name, scorer in scorers.items():
                score_value = float(scorer(best_pipeline, X_test, y_test))
                fold_scores[metric_name].append(score_value)
                fold_result[f"test_{metric_name}"] = score_value

            fold_history[model_name].append(fold_result)

        # Final fit on all data
        final_pipeline = _build_pipeline(
            feature_selection,
            preprocessor,
            estimator,
            scoring.get(primary_metric),
            inner_cv.get_n_splits(),
        )
        final_param_grid = _resolve_search_space(model_name, config, X_df.shape[1])
        if feature_selection != "rfe" and "rfe__n_features_to_select" in final_param_grid:
            final_param_grid = {
                k: v
                for k, v in final_param_grid.items()
                if k != "rfe__n_features_to_select"
            }

        final_search = GridSearchCV(
            final_pipeline,
            param_grid=final_param_grid,
            scoring=scoring,
            refit=primary_metric,
            cv=inner_cv,
            n_jobs=-1,
        )
        final_search.fit(X_df, y_series)
        best_pipeline = _finalize_rfecv_pipeline(final_search.best_estimator_, X_df, y_series, config, scoring.get(primary_metric))
        if label_encoder is not None:
            setattr(best_pipeline, "label_encoder_", label_encoder)
        best_estimators[model_name] = best_pipeline

        final_feature_count = _selected_feature_count(feature_selection, best_pipeline)
        if final_feature_count is None:
            final_feature_count = X_df.shape[1]

        final_selected_features = None
        if feature_selection in {"univariate", "rfe", "rfecv"}:
            final_selected_features = _selected_feature_names(
                feature_selection,
                best_pipeline,
                list(X_df.columns),
            )

        summary: Dict[str, Any] = {
            "model": model_name,
            "primary_metric": primary_metric,
            "best_params_full_fit": final_search.best_params_,
            "selected_feature_count": final_feature_count,
            "selected_features": final_selected_features,
        }
        for metric_name, values in fold_scores.items():
            summary[f"mean_{metric_name}"] = float(np.mean(values))
            summary[f"std_{metric_name}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        records.append(summary)

    leaderboard = pd.DataFrame(records).sort_values(
        by=f"mean_{primary_metric}", ascending=False
    ).reset_index(drop=True)
    leaderboard["fold_details"] = leaderboard["model"].map(fold_history)
    try:
        run_dir = log_training_run(
            config=config,
            leaderboard=leaderboard,
            trained_models=best_estimators,
        )
        leaderboard.attrs["run_dir"] = str(run_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to log training run: {exc}") from exc
    return leaderboard, best_estimators
