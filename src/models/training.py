from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import RFE, RFECV, SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, get_scorer, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from src.models.metrics import scoring_map
from src.utils.config import PipelineConfig
from src.utils.run_logger import log_training_run
from src.features.preprocess import build_fold_preprocessor
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
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _ensure_series(y) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Nested CV currently supports a single target column.")
        return y.iloc[:, 0]
    if isinstance(y, pd.Series):
        return y
    return pd.Series(y)

def _sanitize_param_grid(grid: Dict[str, Any], n_features: int) -> Dict[str, Any]:
    sanitized = {}

    for key, raw_values in grid.items():
        # Distribution case
        if hasattr(raw_values, "rvs"):
            sanitized[key] = raw_values
            continue

        # Normal list-or-single value case
        if isinstance(raw_values, (list, tuple)):
            values = list(raw_values)
        else:
            values = [raw_values]

        # Handle RFE explicitly
        if key == "rfe__n_features_to_select":
            filtered = []
            for v in values:
                if v is None or v == "all":
                    filtered.append(n_features)
                elif isinstance(v, int) and 1 <= v <= n_features:
                    filtered.append(v)
            sanitized[key] = filtered
            continue

        sanitized[key] = values

    return sanitized


def _estimator_supports_feature_selection(
    feature_selection: str,
    estimator: BaseEstimator,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
) -> bool:
    # Always allow usage; fallback happens later if sklearn raises
    if feature_selection in {"none", "univariate"}:
        return True
    if feature_selection in {"rfe", "rfecv"}:
        return True
    return False



def _resolve_search_space(
    model_name: str, config: PipelineConfig, n_features: int
) -> Dict[str, Any]:
    # Only use user-provided search spaces
    user_spaces = getattr(config, "search_spaces", {}) or {}

    # Return the grid if provided, otherwise empty dict
    raw_grid = user_spaces.get(model_name, {})

    return _sanitize_param_grid(raw_grid, n_features)


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
        return "None"

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


def _build_pipeline(feature_selection: str, fold_preprocessor, estimator, scoring: str | None = None, cv_splits: int = 5):
    steps = [("preprocess", clone(fold_preprocessor))]
    if feature_selection == "univariate":
        steps.append(("select", SelectKBest(score_func=partial(mutual_info_classif, random_state=PipelineConfig.random_state))))
    elif feature_selection == "rfe":
        steps.append(("rfe", RFE(estimator=clone(estimator))))
    elif feature_selection == "rfecv":
        steps.append(("rfecv", RFECV(estimator=clone(estimator), cv=cv_splits, scoring=scoring, step=1, min_features_to_select=1)))
    # "none" just skips feature selection
    steps.append(("estimator", clone(estimator)))
    return Pipeline(steps=steps)



def _finalize_rfecv_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig,
    scoring_code: str | None,
    *,
    batch_labels
):
    """
    Finalize an RFECV/RFE pipeline by freezing feature selection into a
    FixedFeatureSelector, then refitting preprocess + selector + estimator.

    batch_labels is REQUIRED because preprocessors like CombatCorrector
    need metadata during refit.
    """

    # If no RFECV/RFE step exists → nothing to finalize
    if "rfecv" not in pipeline.named_steps and "rfe" not in pipeline.named_steps:
        return pipeline

    # Extract selector depending on mode
    selector = pipeline.named_steps.get("rfecv") or pipeline.named_steps.get("rfe")
    support = getattr(selector, "support_", None)

    # If selector failed or did not compute support → return raw pipeline
    if support is None:
        return pipeline

    support = np.asarray(support, dtype=bool)

    # Pull relevant steps from original pipeline
    pre = pipeline.named_steps.get("preprocess")
    est = pipeline.named_steps.get("estimator")

    # Build a clean pipeline with frozen selector
    finalized_steps = []
    if pre is not None:
        finalized_steps.append(("preprocess", clone(pre)))

    finalized_steps.append(("feature_select", FixedFeatureSelector(support)))

    if est is not None:
        finalized_steps.append(("estimator", clone(est)))

    finalized = Pipeline(finalized_steps)

    # CRITICAL: pass batch_labels for CombatCorrector
    finalized.fit(X, y, batch_labels=batch_labels)

    finalized.selected_support_ = support
    finalized.rfecv_cv_results_ = getattr(selector, "cv_results_", None)

    return finalized


def _get_classes_from_pipeline(p):
    """Extract classes_ from pipeline or estimator."""
    if hasattr(p, "classes_"):
        return p.classes_
    if hasattr(p, "named_steps") and "estimator" in p.named_steps:
        est = p.named_steps["estimator"]
        return getattr(est, "classes_", None)
    return None


def _compute_roc_payload(best_pipeline, X_test, y_test, classes, task_type, test_batches):
    """Unified ROC computation for binary + multiclass."""
    from sklearn.metrics import roc_curve
    has_proba = hasattr(best_pipeline, "predict_proba")
    has_df = hasattr(best_pipeline, "decision_function")

    scores = None

    # get model scores
    if has_proba:
        scores = best_pipeline.predict_proba(X_test, batch_labels=test_batches)
    elif has_df:
        scores = best_pipeline.decision_function(X_test, batch_labels=test_batches)
    else:
        return None

    scores = np.asarray(scores)

    # ---- BINARY ROC ----
    if task_type == "binary":
        if scores.ndim == 2 and scores.shape[1] > 1:
            # positive class index
            class_list = list(classes) if classes is not None else [0, 1]
            pos_idx = class_list.index(1) if 1 in class_list else scores.shape[1] - 1
            pos_scores = scores[:, pos_idx]
        else:
            pos_scores = scores.ravel()

        try:
            fpr, tpr, thr = roc_curve(y_test, pos_scores)
            return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
        except ValueError:
            return None

    # ---- MULTICLASS ROC ----
    payload = {"per_class": []}
    y_values = y_test.to_numpy()
    if classes is None:
        classes = np.arange(scores.shape[1])

    for idx, c in enumerate(classes):
        if idx >= scores.shape[1]:
            continue
        binary = (y_values == c).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        try:
            fpr, tpr, thr = roc_curve(binary, scores[:, idx])
            payload["per_class"].append({
                "class_label": c,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thr.tolist(),
            })
        except ValueError:
            pass

    return payload if payload["per_class"] else None


def _fit_with_search(X, y, batches, config, model_feature_selection,
                     estimator, primary_metric, inner_cv, scoring, model_name=None):
    """
    Performs hyperparameter search with optional RFE/RFECV.
    If RFE/RFECV fails (estimator doesn't support coef_ or errors occur),
    it gracefully falls back to feature_selection='none'.
    """

    def run_search(feature_selection_mode):
        """Internal helper to build & run a search once."""
        preprocessor = build_fold_preprocessor(batch_labels=batches)

        pipeline = _build_pipeline(
            feature_selection_mode,
            preprocessor,
            estimator,
            scoring.get(primary_metric),
            inner_cv.get_n_splits(),
        )

        param_grid = _resolve_search_space(model_name, config, X.shape[1])

        # Remove RFE params if this run does not use RFE
        if feature_selection_mode != "rfe":
            param_grid.pop("rfe__n_features_to_select", None)

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            scoring=scoring,
            refit=primary_metric,
            cv=inner_cv,
            n_jobs=-1,
            random_state=config.random_state
        )
        search.fit(X, y, batch_labels=batches)
        return search

    # =========================================
    # Try ORIGINAL feature selection first
    # =========================================
    try:
        search = run_search(model_feature_selection)

    except Exception as e:
        print(
            f"[FEATURE] {model_feature_selection.upper()} FAILED for "
            f"{estimator.__class__.__name__}. Falling back to 'none'. "
            f"Error={type(e).__name__}: {e}"
        )

        # Retry with no feature selection
        search = run_search("none")
        model_feature_selection = "none"

    # =========================================
    # Finalize pipeline
    # =========================================

    best_pipeline = _finalize_rfecv_pipeline(
        search.best_estimator_, X, y, config, scoring.get(primary_metric),batch_labels=batches
    )

    feat_count = _selected_feature_count(model_feature_selection, best_pipeline)
    if feat_count is None:
        feat_count = X.shape[1]

    return best_pipeline, search.best_params_, feat_count


def nested_cross_validate_models(
    models: Dict[str, BaseEstimator],
    X,
    y,
    config: PipelineConfig,
):
    print("\n========== NESTED CROSS-VALIDATION ==========")

    scoring = scoring_map(config.task_type)
    primary_metric = next(iter(scoring))
    print(f"[INIT] Primary metric: {primary_metric}")

    X_df = _ensure_frame(X)
    y_series = _ensure_series(y)

    # Batches
    batches_all = X_df.index.get_level_values("batch")
    print(f"[DATA] X={X_df.shape}, y={y_series.shape}, batches={batches_all.unique()}")

    # Encode labels
    le = LabelEncoder()
    y_series = pd.Series(le.fit_transform(y_series), index=y_series.index)

    outer_cv = build_outer_cv(config)
    inner_cv = build_inner_cv(config)

    feature_selection = getattr(config, "feature_selection", "none")
    selection_cache = {}

    results = []
    fold_history = {}
    final_models = {}

    for model_name, estimator in models.items():
        print(f"\n===== MODEL: {model_name} =====")

        # determine if estimator supports RFE/RFECV
        if (model_name, feature_selection) not in selection_cache:
            ok = _estimator_supports_feature_selection(
                feature_selection, estimator, X_df, y_series
            )
            selection_cache[(model_name, feature_selection)] = ok

        supports = selection_cache[(model_name, feature_selection)]
        true_feature_sel = feature_selection if supports else "none"
        print(f"[FEATURE] {model_name}: using feature_selection={true_feature_sel}")

        fold_history[model_name] = []
        fold_scores = {m: [] for m in scoring}

        # ---------- OUTER LOOP ----------
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_df, y_series)):
            print(f"\n--- OUTER FOLD {fold_idx+1}/{outer_cv.n_splits} ---")

            X_train = X_df.iloc[train_idx]
            X_test = X_df.iloc[test_idx]
            y_train = y_series.iloc[train_idx]
            y_test = y_series.iloc[test_idx]

            train_batches = X_train.index.get_level_values("batch")
            test_batches = X_test.index.get_level_values("batch")

            # Run inner CV search 
            best_pipe, best_params, feat_count = _fit_with_search(
                X_train, y_train, train_batches,
                config, true_feature_sel,
                estimator, primary_metric,
                inner_cv, scoring, model_name=model_name,
            )
            # Record selected feature names
            fold_selected_names = _selected_feature_names(
                true_feature_sel,
                best_pipe,
                X_train.columns.tolist()
            )

            fold = {"outer_fold": fold_idx, "best_params": best_params}
            fold["selected_feature_names"] = fold_selected_names
            fold["selected_feature_count"] = feat_count

            print(f"[SEARCH] Best params: {best_params}")
            print(f"[FEATURE] Selected {feat_count} features")

            # Predictions
            try:
                y_pred = best_pipe.predict(X_test, batch_labels=test_batches)
            except Exception:
                y_pred = None

            # Confusion matrix
            if y_pred is not None:
                cm = confusion_matrix(y_test, y_pred)
                fold["confusion_matrix"] = cm.tolist()

            # ROC
            classes = _get_classes_from_pipeline(best_pipe)
            roc_payload = _compute_roc_payload(
                best_pipe, X_test, y_test,
                classes, config.task_type, test_batches
            )
            if roc_payload:
                fold["roc_curve"] = roc_payload

            # Compute metrics
            if y_pred is not None:
                for metric_name, scorer in scoring.items():
                    score = get_scorer(scorer)._score_func(
                        y_test,
                        y_pred,
                        **get_scorer(scorer)._kwargs
                    )
                    fold_scores[metric_name].append(score)
                    fold[f"test_{metric_name}"] = float(score)

            fold_history[model_name].append(fold)

        # ---------- FINAL MODEL ON ALL DATA ----------
        print("[FINAL] Training final model on all data...")
        final_pipe, final_params, final_feat_count = _fit_with_search(
            X_df, y_series, batches_all,
            config, true_feature_sel,
            estimator, primary_metric,
            inner_cv, scoring, model_name=model_name, 
        )
        final_pipe.label_encoder_ = le
        final_models[model_name] = final_pipe

        # Extract final selected feature names
        input_columns = list(X_df.columns)

        if true_feature_sel in {"univariate", "rfe", "rfecv"}:
            final_selected_names = _selected_feature_names(
                true_feature_sel,
                final_pipe,
                input_columns
            )
        else:
            final_selected_names = "All"

        # Write summary entry
        summary = {
            "model": model_name,
            "primary_metric": primary_metric,
            "best_params_full_fit": final_params,
            "selected_feature_count": final_feat_count,
            "selected_features": final_selected_names,
            "feature_selection": true_feature_sel,
        }

        for metric_name, values in fold_scores.items():
            summary[f"mean_{metric_name}"] = np.mean(values)
            summary[f"std_{metric_name}"] = np.std(values) if len(values)>1 else 0

        results.append(summary)

    # ---------- Produce leaderboard ----------
    leaderboard = pd.DataFrame(results).sort_values(
        by=f"mean_{primary_metric}", ascending=False
    ).reset_index(drop=True)
    leaderboard["fold_details"] = leaderboard["model"].map(fold_history)

    print("\n========== NESTED CV COMPLETED ==========")

    # ---- Log training run ----
    try:
        run_dir = log_training_run(
            config=config,
            leaderboard=leaderboard,
            trained_models=final_models,
        )
        leaderboard.attrs["run_dir"] = str(run_dir)
        print(f"[LOG] Training run logged to: {run_dir}")
    except Exception as e:
        print(f"[LOG] Failed to log training run: {type(e).__name__}: {e}")

    return leaderboard, final_models
