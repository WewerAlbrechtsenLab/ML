from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
import pandas as pd
import json
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import confusion_matrix, get_scorer, roc_curve
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV

from ml.models.metrics import scoring_map
from ml.utils.config import PipelineConfig
from ml.utils.run_logger import log_training_run
from ml.features.preprocess import build_fold_preprocessor


# -----------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------

def _ensure_frame(X):
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

def _ensure_series(y):
    if isinstance(y, pd.Series):
        return y
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Only single target supported.")
        return y.iloc[:, 0]
    return pd.Series(y)

def write_leaderboard(results, primary_metric, fold_history, output_dir, prefix):
    """
    Safe leaderboard writer:
    - Updates model-by-model.
    - Never crashes if metrics are missing.
    - Overwrites one stable file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = pd.DataFrame(results)

    metric_col = f"mean_{primary_metric}"
    if metric_col in leaderboard.columns:
        leaderboard = leaderboard.sort_values(
            by=metric_col,
            ascending=False,
            na_position="last"
        )

    leaderboard = leaderboard.reset_index(drop=True)

    leaderboard["fold_details"] = leaderboard["model"].map(fold_history)

    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}_folds.json"

    try:
        leaderboard.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(fold_history, indent=2))
        print(f"[LOG] Leaderboard updated: {csv_path}")
    except Exception as e:
        print(f"[WRITE ERROR] Could not write leaderboard: {e}")

    return csv_path, json_path

# -----------------------------------------------------------
# ROC computation
# -----------------------------------------------------------

def compute_roc(estimator, X, y, classes, task_type):
    has_proba = hasattr(estimator, "predict_proba")
    has_df = hasattr(estimator, "decision_function")

    if has_proba:
        scores = estimator.predict_proba(X)
    elif has_df:
        scores = estimator.decision_function(X)
    else:
        return None

    scores = np.asarray(scores)

    # Binary
    if task_type == "binary":
        if scores.ndim == 2:
            pos_idx = list(classes).index(1) if 1 in classes else 1
            pos_scores = scores[:, pos_idx]
        else:
            pos_scores = scores

        fpr, tpr, thr = roc_curve(y, pos_scores)
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}

    # Multiclass
    payload = {"per_class": []}
    y_arr = y.to_numpy()

    for idx, cls in enumerate(classes):
        binary = (y_arr == cls).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue

        fpr, tpr, thr = roc_curve(binary, scores[:, idx])
        payload["per_class"].append(
            dict(class_label=cls, fpr=fpr.tolist(), tpr=tpr.tolist(), thresholds=thr.tolist())
        )

    return payload if payload["per_class"] else None


# -----------------------------------------------------------
# RFECV (ONE-TIME) using tuned estimator
# -----------------------------------------------------------

def run_rfecv_once(X, y, tuned_estimator, scoring, inner_cv, batches=None):
    pre = None                                                                      ## Edit if preprocessor is needed
    # pre = build_fold_preprocessor(batch_labels=batches)

    if pre is not None:
        X_local = clone(pre).fit_transform(X, y, batch_labels=batches)
    else:
        X_local = X

    rfecv = RFECV(
        estimator=clone(tuned_estimator),
        cv=inner_cv,
        scoring=scoring,
        step=1,
        min_features_to_select=1,
    )
    rfecv.fit(X_local, y)

    return np.asarray(rfecv.support_, dtype=bool)


# -----------------------------------------------------------
# Hyperparameter search
# -----------------------------------------------------------

def run_hp_search(X, y, estimator, config, scoring, primary_metric, inner_cv, model_name):
    param_grid = config.search_spaces.get(model_name, {})

    search = RandomizedSearchCV(
        clone(estimator),
        param_distributions=param_grid,
        scoring=scoring,
        refit=primary_metric,
        cv=inner_cv,
        n_jobs=-1,
        random_state=config.random_state,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


# -----------------------------------------------------------
# Main training: HP-first, RFECV-second
# -----------------------------------------------------------

def nested_cross_validate_models(models, X, y, config: PipelineConfig):
    print("\n========== NESTED CV: HP-FIRST + RFECV-SECOND ==========")

    X = _ensure_frame(X)
    y = _ensure_series(y)

    scoring = scoring_map(config.task_type)
    primary_metric = next(iter(scoring))

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index)

    outer_cv = StratifiedKFold(config.outer_splits, shuffle=True, random_state=config.random_state)
    inner_cv = StratifiedKFold(config.inner_splits, shuffle=True, random_state=config.random_state)

    results = []
    fold_history = {}
    final_models = {}

    for model_name, base_estimator in models.items():
        print(f"\n===== MODEL: {model_name} =====")
        fold_history[model_name] = []
        fold_scores = {m: [] for m in scoring}

        try:
            # -------- OUTER LOOP --------
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                print(f"\n--- OUTER FOLD {fold_idx+1}/{config.outer_splits} ---")

                Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
                ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

                tuned_estimator, best_params = run_hp_search(
                    Xtr, ytr,
                    base_estimator,
                    config,
                    scoring,
                    primary_metric,
                    inner_cv,
                    model_name,
                )
                print(f"[HP-SEARCH] Best params: {best_params}")

                mask = run_rfecv_once(
                    Xtr, ytr,
                    tuned_estimator,
                    scoring[primary_metric],
                    inner_cv,
                    batches=None,
                )

                selected = X.columns[mask].tolist()
                print(f"[RFECV] Selected {mask.sum()} features")

                Xtr_sel = Xtr.loc[:, mask]
                Xte_sel = Xte.loc[:, mask]

                tuned_estimator = clone(tuned_estimator).fit(Xtr_sel, ytr)
                y_pred = tuned_estimator.predict(Xte_sel)

                fold = dict(
                    outer_fold=fold_idx,
                    best_params=best_params,
                    selected_features=selected,
                    selected_feature_count=int(mask.sum()),
                    confusion_matrix=confusion_matrix(yte, y_pred).tolist(),
                )

                classes = tuned_estimator.classes_
                roc_payload = compute_roc(tuned_estimator, Xte_sel, yte,
                                          classes, config.task_type)
                if roc_payload:
                    fold["roc_curve"] = roc_payload

                for m, scorer_code in scoring.items():
                    scorer = get_scorer(scorer_code)
                    score = scorer._score_func(yte, y_pred, **scorer._kwargs)
                    fold_scores[m].append(score)
                    fold[f"test_{m}"] = float(score)

                fold_history[model_name].append(fold)

            # -------- FINAL MODEL --------
            tuned_full, final_params = run_hp_search(
                X, y,
                base_estimator,
                config,
                scoring,
                primary_metric,
                inner_cv,
                model_name,
            )
            print(f"[FINAL] Best params: {final_params}")

            full_mask = run_rfecv_once(
                X, y,
                tuned_full,
                scoring[primary_metric],
                inner_cv,
                batches=None,
            )

            selected_full = X.columns[full_mask].tolist()
            print(f"[FINAL RFECV] Selected {full_mask.sum()} features")

            final_model = clone(tuned_full).fit(X.loc[:, full_mask], y)
            final_model.label_encoder_ = le
            final_models[model_name] = (final_model, full_mask)

            summary = {
                "model": model_name,
                "primary_metric": primary_metric,
                "best_params_full_fit": final_params,
                "selected_feature_count": int(full_mask.sum()),
                "selected_features": selected_full,
            }

            for m, values in fold_scores.items():
                summary[f"mean_{m}"] = float(np.mean(values))
                summary[f"std_{m}"] = float(np.std(values)) if len(values)>1 else 0.0

            results.append(summary)

            # ---- update leaderboard after every model ----
            write_leaderboard(
                results, primary_metric, fold_history,
                config.output_dir, prefix="leaderboard_partial"
            )

        except Exception as e:
            print(f"[ERROR] Model '{model_name}' failed: {type(e).__name__}: {e}")

            results.append({
                "model": model_name,
                "primary_metric": primary_metric,
                "error": f"{type(e).__name__}: {e}",
            })

            write_leaderboard(
                results, primary_metric, fold_history,
                config.output_dir, prefix="leaderboard_partial"
            )
            continue

    # -------- FINAL LEADERBOARD --------
    write_leaderboard(
        results, primary_metric, fold_history,
        config.output_dir, prefix="leaderboard_final"
    )

    return pd.DataFrame(results), final_models
