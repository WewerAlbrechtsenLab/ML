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
from src.features.preprocess2 import build_fold_preprocessor
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


def _is_distribution_candidate(value: Any) -> bool:
    return hasattr(value, "rvs") and callable(getattr(value, "rvs"))


def _sanitize_param_grid(
    grid: Dict[str, Any], n_features: int, include_rfe: bool
) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, raw_values in grid.items():
        if _is_distribution_candidate(raw_values):
            values = raw_values
        elif isinstance(raw_values, (list, tuple)):
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
        elif _is_distribution_candidate(values):
            sanitized[key] = values
        else:
            sanitized[key] = values

    if include_rfe and "rfe__n_features_to_select" not in sanitized:
        sanitized["rfe__n_features_to_select"] = _default_rfe_values(n_features)
    return sanitized


def _estimator_supports_feature_selection(
    feature_selection: str,
    estimator: BaseEstimator,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
) -> bool:
    """Return True when the estimator can drive the requested feature selection."""
    if feature_selection in {"none", "univariate"}:
        return True
    if feature_selection not in {"rfe", "rfecv"}:
        return False

    try:
        steps = []
        steps.append(("estimator", clone(estimator)))
        probe = Pipeline(steps=steps)
    except Exception:
        return False

    try:
        probe.fit(X_sample, y_sample)
    except Exception:
        return False

    fitted = probe.named_steps.get("estimator")
    if fitted is None:
        return False

    return any(
        hasattr(fitted, attr) for attr in ("feature_importances_", "coef_")
    )


def _resolve_search_space(
    model_name: str, config: PipelineConfig, n_features: int
) -> Dict[str, Any]:
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

    raw_labels = pd.Index(y_series.unique())
    if config.task_type == "binary":
        if len(raw_labels) != 2:
            raise ValueError(
                f"Binary classification requires 2 classes, got {len(raw_labels)}: {raw_labels}"
            )

    label_encoder = LabelEncoder()
    y_series = pd.Series(
        label_encoder.fit_transform(y_series),
        index=y_series.index
    )

    # ---- Build CV objects  ----
    outer_cv = build_outer_cv(config)
    inner_cv = build_inner_cv(config)

    scorers = {name: get_scorer(code) for name, code in scoring.items()}

    records: List[Dict[str, Any]] = []
    fold_history: Dict[str, List[Dict[str, Any]]] = {}
    best_estimators: Dict[str, BaseEstimator] = {}

    feature_selection = getattr(config, "feature_selection", "none")
    selection_support_cache: Dict[Tuple[str, str], bool] = {}

    for model_name, estimator in models.items():
        model_feature_selection = feature_selection
        cache_key = (model_name, model_feature_selection)
        supports_selection = selection_support_cache.get(cache_key)
        if supports_selection is None:
            supports_selection = _estimator_supports_feature_selection(
                model_feature_selection,
                estimator,
                X_df,
                y_series,
            )
            selection_support_cache[cache_key] = supports_selection
        if not supports_selection:
            model_feature_selection = "none"

        fold_scores: Dict[str, List[float]] = {metric: [] for metric in scoring}
        fold_history[model_name] = []

        # ===================== OUTER CV LOOP ===================== #
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_df, y_series)):
            X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
            y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]

            # Extract fold-specific batch labels for ComBat (from MultiIndex level "batch")
            train_batches = X_train.index.get_level_values("batch")
            test_batches = X_test.index.get_level_values("batch")

            # Build NEW fold-specific preprocessor
            fold_preprocessor = build_fold_preprocessor(batch_labels=train_batches)

            # Build full ML pipeline for inner CV (preprocess + feature selection + estimator)
            pipeline = _build_pipeline(
                model_feature_selection,
                fold_preprocessor,
                estimator,
                scoring.get(primary_metric),
                inner_cv.get_n_splits(),
            )

            # Resolve model-specific search space
            param_grid = _resolve_search_space(model_name, config, X_train.shape[1])
            if model_feature_selection != "rfe" and "rfe__n_features_to_select" in param_grid:
                param_grid = {
                    k: v for k, v in param_grid.items() if k != "rfe__n_features_to_select"
                }

            # Inner CV hyperparameter search
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                scoring=scoring,
                refit=primary_metric,
                cv=inner_cv,
                n_jobs=-1,
            )

            # Pass batch_labels only to the estimator pipeline; metadata routing will
            # send it to CombatCorrector.fit / transform, not to scorers.
            search.fit(
                X_train,
                y_train,
                batch_labels=train_batches,
            )

            best_pipeline = search.best_estimator_
            

            best_pipeline = _finalize_rfecv_pipeline(
                best_pipeline, X_train, y_train, config, scoring.get(primary_metric)
            )

            fold_result: Dict[str, Any] = {
                "outer_fold": fold_idx,
                "best_params": search.best_params_,
            }

            # ===================== PREDICTIONS ON OUTER TEST ===================== #
            classes = getattr(best_pipeline, "classes_", None)
            if classes is None and hasattr(best_pipeline, "named_steps"):
                estimator_step = best_pipeline.named_steps.get("estimator")
                classes = getattr(estimator_step, "classes_", None)

            try:
                y_pred = best_pipeline.predict(
                    X_test,
                    batch_labels=test_batches,
                )
            except Exception:
                y_pred = None
            else:
                # Confusion matrix
                if label_encoder is not None:
                    label_range = np.arange(len(label_encoder.classes_))
                    cm = confusion_matrix(y_test, y_pred, labels=label_range)
                    cm_labels = label_encoder.inverse_transform(label_range).tolist()
                else:
                    if classes is not None:
                        label_values = list(classes)
                    else:
                        merged = np.concatenate([y_train.to_numpy(), y_test.to_numpy()])
                        label_values = list(pd.Index(np.unique(merged)))
                    cm = confusion_matrix(y_test, y_pred, labels=label_values)
                    cm_labels = label_values
                cm_labels = [
                    label.item() if isinstance(label, np.generic) else label for label in cm_labels
                ]
                fold_result["confusion_matrix"] = cm.tolist()
                fold_result["confusion_matrix_labels"] = cm_labels

            # ===================== ROC CURVES (OPTIONAL) ===================== #
            roc_curve_payload = None
            has_predict_proba = hasattr(best_pipeline, "predict_proba")
            has_decision_function = hasattr(best_pipeline, "decision_function")

            if config.task_type == "binary" and has_predict_proba:
                proba = None
                try:
                    proba = best_pipeline.predict_proba(
                        X_test,
                        batch_labels=test_batches,
                    )
                except Exception:
                    proba = None

                if proba is not None:
                    proba = np.asarray(proba)
                    if proba.ndim == 2 and proba.shape[1] > 1:
                        if classes is None and hasattr(best_pipeline, "named_steps"):
                            estimator_step = best_pipeline.named_steps.get("estimator")
                            classes = getattr(estimator_step, "classes_", None)
                        class_list = list(classes) if classes is not None else None
                        pos_index = -1
                        if class_list is not None and 1 in class_list:
                            pos_index = class_list.index(1)
                        if pos_index < 0:
                            pos_index = proba.shape[1] - 1
                        pos_scores = proba[:, pos_index]
                    else:
                        pos_scores = proba.ravel()
                    try:
                        fpr, tpr, thresholds = roc_curve(y_test, pos_scores)
                    except ValueError:
                        roc_curve_payload = None
                    else:
                        roc_curve_payload = {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist(),
                            "thresholds": thresholds.tolist(),
                        }

            elif (
                config.task_type == "multiclass"
                and (has_predict_proba or has_decision_function)
            ):
                class_scores = None
                try:
                    if has_predict_proba:
                        class_scores = best_pipeline.predict_proba(
                            X_test,
                            batch_labels=test_batches,
                        )
                    else:
                        class_scores = best_pipeline.decision_function(
                            X_test,
                            batch_labels=test_batches,
                        )
                except Exception:
                    class_scores = None

                if class_scores is not None:
                    scores = np.asarray(class_scores)
                    if scores.ndim == 1:
                        scores = scores.reshape(-1, 1)
                    if classes is None and hasattr(best_pipeline, "named_steps"):
                        estimator_step = best_pipeline.named_steps.get("estimator")
                        classes = getattr(estimator_step, "classes_", None)
                    if classes is None:
                        classes = np.arange(scores.shape[1])
                    class_list = list(classes)
                    per_class_curves: List[Dict[str, Any]] = []
                    y_test_values = y_test.to_numpy()
                    for idx, encoded_label in enumerate(class_list):
                        if scores.shape[1] <= idx:
                            continue
                        binary_targets = (y_test_values == encoded_label).astype(int)
                        # Need at least one positive and one negative sample to compute ROC.
                        if binary_targets.sum() == 0 or binary_targets.sum() == binary_targets.size:
                            continue
                        try:
                            fpr, tpr, thresholds = roc_curve(binary_targets, scores[:, idx])
                        except ValueError:
                            continue
                        display_label = encoded_label
                        if label_encoder is not None:
                            display_label = label_encoder.inverse_transform([encoded_label])[0]
                        if isinstance(display_label, np.generic):
                            display_label = display_label.item()
                        per_class_curves.append(
                            {
                                "class_label": display_label,
                                "fpr": fpr.tolist(),
                                "tpr": tpr.tolist(),
                                "thresholds": thresholds.tolist(),
                            }
                        )
                    if per_class_curves:
                        roc_curve_payload = {"per_class": per_class_curves}
            if roc_curve_payload is not None:
                fold_result["roc_curve"] = roc_curve_payload

            # ===================== FEATURE SELECTION META ===================== #
            feature_count = _selected_feature_count(model_feature_selection, best_pipeline)
            if feature_count is None:
                feature_count = X_train.shape[1]
            fold_result["selected_feature_count"] = feature_count

            selected_features = "All"
            if model_feature_selection in {"univariate", "rfe", "rfecv"}:
                selected_features = _selected_feature_names(
                    model_feature_selection,
                    best_pipeline,
                    list(X_train.columns),
                )
            fold_result["selected_features"] = selected_features
            fold_result["feature_selection"] = model_feature_selection

            # ===================== METRICS USING SCORER OBJECTS ===================== #
            if y_pred is not None:
                for metric_name, scorer in scorers.items():
                    # Figure out what the scorer expects: predict, predict_proba, or decision_function
                    response_method = getattr(scorer, "_response_method", "predict")

                    if response_method == "predict":
                        y_score = y_pred

                    elif response_method == "predict_proba" and has_predict_proba:
                        # For probability-based metrics (e.g. roc_auc), mimic sklearn's internal behaviour
                        proba = best_pipeline.predict_proba(
                            X_test,
                            batch_labels=test_batches,
                        )
                        proba = np.asarray(proba)
                        # Binary: pass proba for the positive class
                        if config.task_type == "binary" and proba.ndim == 2 and proba.shape[1] > 1:
                            class_list = list(classes) if classes is not None else None
                            pos_index = -1
                            if class_list is not None and 1 in class_list:
                                pos_index = class_list.index(1)
                            if pos_index < 0:
                                pos_index = proba.shape[1] - 1
                            y_score = proba[:, pos_index]
                        else:
                            y_score = proba

                    elif response_method == "decision_function" and has_decision_function:
                        y_score = best_pipeline.decision_function(
                            X_test,
                            batch_labels=test_batches,
                        )
                    else:
                        # Fallback: just use predictions
                        y_score = y_pred

                    score_value = scorer._sign * scorer._score_func(
                        y_test,
                        y_score,
                        **scorer._kwargs,
                    )
                    score_value = float(score_value)
                    fold_scores[metric_name].append(score_value)
                    fold_result[f"test_{metric_name}"] = score_value

            fold_history[model_name].append(fold_result)

        print(f"Completed nested CV for model '{model_name}'.")

        # ===================== FINAL FIT ON ALL DATA ===================== #
        all_batches = X_df.index.get_level_values("batch")
        global_preprocessor = build_fold_preprocessor(batch_labels=all_batches)

        final_pipeline = _build_pipeline(
            model_feature_selection,
            global_preprocessor,
            estimator,
            scoring.get(primary_metric),
            inner_cv.get_n_splits(),
        )

        final_param_grid = _resolve_search_space(model_name, config, X_df.shape[1])
        if model_feature_selection != "rfe" and "rfe__n_features_to_select" in final_param_grid:
            final_param_grid = {
                k: v
                for k, v in final_param_grid.items()
                if k != "rfe__n_features_to_select"
            }

        final_search = RandomizedSearchCV(
            final_pipeline,
            param_distributions=final_param_grid,
            scoring=scoring,
            refit=primary_metric,
            cv=inner_cv,
            n_jobs=-1,
        )
        final_search.fit(
            X_df,
            y_series,
            batch_labels=all_batches,
        )

        best_pipeline = _finalize_rfecv_pipeline(
            final_search.best_estimator_, X_df, y_series, config, scoring.get(primary_metric)
        )
        if label_encoder is not None:
            setattr(best_pipeline, "label_encoder_", label_encoder)
        best_estimators[model_name] = best_pipeline

        final_feature_count = _selected_feature_count(model_feature_selection, best_pipeline)
        if final_feature_count is None:
            final_feature_count = X_df.shape[1]

        final_selected_features = "All"
        if model_feature_selection in {"univariate", "rfe", "rfecv"}:
            final_selected_features = _selected_feature_names(
                model_feature_selection,
                best_pipeline,
                list(X_df.columns),
            )

        summary: Dict[str, Any] = {
            "model": model_name,
            "primary_metric": primary_metric,
            "best_params_full_fit": final_search.best_params_,
            "selected_feature_count": final_feature_count,
            "selected_features": final_selected_features,
            "feature_selection": model_feature_selection,
        }
        for metric_name, values in fold_scores.items():
            summary[f"mean_{metric_name}"] = float(np.mean(values))
            summary[f"std_{metric_name}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        records.append(summary)

    print("Nested cross-validation completed for all models.")
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
        print(f"Training run logged to: {run_dir}")
    except Exception as exc:
        raise RuntimeError(f"Failed to log training run: {exc}") from exc
    return leaderboard, best_estimators
