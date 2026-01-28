from __future__ import annotations

from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, get_scorer, roc_curve
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV

from ml.models.metrics import scoring_map
from ml.utils.config import PipelineConfig
from ml.models.visualization import plot_rfecv_curve
from MSprocessing.stats.models import run_linear_model


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

def save_feature_selection_json(feature_selection_dict, output_dir, filename="feature_selection_linear_model.json"):
    json_ready = {}

    for model, fold_dict in feature_selection_dict.items():
        json_ready[model] = {}
        for fold_key, proteins in fold_dict.items():
            # convert sets → sorted lists (JSON-safe)
            json_ready[model][fold_key] = sorted(list(proteins))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / filename
    with open(out_path, "w") as f:
        json.dump(json_ready, f, indent=2)

    print(f"[SAVED] Feature selection JSON written to: {out_path}")

    return out_path

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
# RFECV 
# -----------------------------------------------------------

def run_rfecv_once(X, y, tuned_estimator, scoring, inner_cv, batches=None):
    pre = None
    if pre is not None:
        X_local = clone(pre).fit_transform(X, y, batch_labels=batches)
    else:
        X_local = X
    try:
    #if hasattr(tuned_estimator, "coef_") or hasattr(tuned_estimator, "feature_importances_"):
        rfecv = RFECV(
            estimator=clone(tuned_estimator),
            cv=inner_cv,
            scoring=scoring,
            step=1,
            min_features_to_select=1,
        )
        rfecv.fit(X_local, y)
        mask = np.asarray(rfecv.support_, dtype=bool)
        return mask, rfecv

    except Exception as e:
        print(f"[RFECV] Skipped for {type(tuned_estimator).__name__}: {type(e).__name__}: {e}")
        mask = np.ones(X.shape[1], dtype=bool)
        return mask, None

def select_mask_within_tolerance(rfecv: RFECV, tolerance: float):
    """
    Pick the SMALLEST feature set whose score is within tolerance of the best.
    """
    scores = rfecv.cv_results_["mean_test_score"]
    stds = rfecv.cv_results_.get("std_test_score", None)

    n_features = rfecv.cv_results_["n_features"]

    best_score = np.max(scores)
    threshold = best_score - tolerance

    # Eligible indices
    eligible = np.where(scores >= threshold)[0]

    # Choose smallest number of features
    best_idx = eligible[np.argmin(n_features[eligible])]

    # Build mask manually
    support = rfecv.support_.copy()

    # Re-run RFECV internals to get exact mask for that feature count
    rfecv.n_features_to_select_ = n_features[best_idx]
    support = rfecv._get_support_mask()

    return support, {
        "best_score": float(best_score),
        "chosen_score": float(scores[best_idx]),
        "chosen_features": int(n_features[best_idx]),
    }

# -----------------------------------------------------------
# Linear Model based Feature Selection 
# -----------------------------------------------------------

def run_linear_models(
    X,
    y,
    inner_cv,
    formula,
    filter_to,
    pval_col="padj",           # "pval" or "padj"
    alpha=0.05,
    coef_threshold=0.0,        # abs(coef) threshold
    keep_k=None,               # keep proteins present in >= keep_k inner folds
    keep_frac=None,            # alternative: keep proteins present in >= keep_frac (0-1)
    max_features=None,          # optional cap after filtering
    mode="evaluation",         # "evaluation" | "deployment" 
    **linear_kwargs,
):
    """
    Confounder-adjusted feature selection using models.run_linear_model. Counts how often each protein passes filters,
    then keeps proteins meeting a stability threshold (>=k folds or >=fraction).

    Parameters
    ----------
    formula : str
        Patsy formula, e.g. "y ~ C(test5, Treatment(reference='control')) + sex + age".
    filter_to : str
        Term to keep from linear model output (e.g. "test5").

    ----------
    Returns:
      mask (np.ndarray bool), selected_features (list), fs_info (dict[str, set[str]])
    """
    X = _ensure_frame(X)
    y = _ensure_series(y)

    # Extract metadata from MultiIndex
    meta = X.index.to_frame(index=False)
    proteome = X.copy()
    proteome.index = meta.index  # align for statsmodels

    if mode == "deployment":
        splits = [(np.arange(len(X)), None)]
    else:
        splits = list(inner_cv.split(X, y))

    n_folds = len(splits)
    fs_info = {}
    # split_info = {}   # for testing
    
    # n_folds = 0

    for fold_id, (tr_i, _) in enumerate(splits, start=1):
        proteome_tr = proteome.iloc[tr_i]
        meta_tr = meta.iloc[tr_i]

        res = run_linear_model(
            proteome=proteome_tr,
            meta=meta_tr,
            formula=formula,
            filter_to=filter_to,
            **linear_kwargs,
        ).copy()

        # ensure protein index
        if res.index.name != "protein" and "protein" in res.columns:
            res = res.set_index("protein")

        passed = (
            (res[pval_col] <= alpha)
            & (res["coef"].abs() > coef_threshold)
        )

        selected = set(res.index[passed])

        fs_info[f"inner_fold_{fold_id}"] = selected
        n_folds += 1

        fs_info[f"inner_fold_{fold_id}"] = selected
        #split_info[f"inner_fold_{fold_id}"] = tr_i  # store indices

        if mode == "evaluation":
            print(
                f"[Linear-FS][Inner {fold_id}] "
                f"passed pval+coef: {len(selected)} features"
            )
        else:
            print(
                f"[Linear-FS][FULL DATA] "
                f"passed pval+coef: {len(selected)} features"
            )


    # -------------------------
    # Selection rule
    # -------------------------
    all_selected = list(fs_info.values())
    counts = {}

    for s in all_selected:
        for p in s:
            counts[p] = counts.get(p, 0) + 1

    if mode == "deployment":
        keep_k_eff = 1
    else:
        if keep_k is None and keep_frac is None:
            keep_k_eff = n_folds
        elif keep_frac is not None:
            keep_k_eff = int(np.ceil(keep_frac * n_folds))
        else:
            keep_k_eff = int(keep_k)


    selected_features = [
        p for p, c in counts.items() if c >= keep_k_eff
    ]
    if mode == "evaluation":
        print(
            f"[Linear-FS] After stability rule: "
            f"{len(selected_features)} features"
        )

    # -------------------------
    # Fallback if empty
    # -------------------------
    if len(selected_features) == 0:
        selected_features = list(X.columns)
        print(
        "[Linear-FS][FALLBACK] "
        "No features met stability rule → using ALL features"
    )

    # -------------------------
    # Optional cap by |coef|
    # -------------------------
    if max_features is not None and len(selected_features) > max_features:
        res_full = run_linear_model(
            proteome=proteome,
            meta=meta,
            formula=formula,
            filter_to=filter_to,
            **linear_kwargs,
        ).copy()

        if res_full.index.name != "protein" and "protein" in res_full.columns:
            res_full = res_full.set_index("protein")

        ranked = (
            res_full.loc[res_full.index.intersection(selected_features)]
            .assign(abscoef=lambda d: d["coef"].abs())
            .sort_values("abscoef", ascending=False)
        )

        selected_features = ranked.head(max_features).index.tolist()
        print(
            f"[Linear-FS] Capped to top {max_features} by |coef|"
        )

    mask = X.columns.isin(selected_features)

    return mask, selected_features, fs_info

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
# Main training
# -----------------------------------------------------------

def nested_cross_validate_models(models, X, y, config: PipelineConfig):
    print("\n========== NESTED CV ==========")

    X = _ensure_frame(X)
    y = _ensure_series(y)

    scoring = scoring_map(config.task_type)
    primary_metric = next(iter(scoring))

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index)

    outer_cv = StratifiedKFold(config.outer_splits, shuffle=True, random_state=config.random_state)
    inner_cv = StratifiedKFold(config.inner_splits, shuffle=True, random_state=config.random_state)

    feature_selection = getattr(config, "feature_selection", "none")

    results = []
    fold_history = {}
    final_models = {}
    info = {}

    figures_dir = Path(config.output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for model_name, base_estimator in models.items():
        print(f"\n===== MODEL: {model_name} =====")
        fold_history[model_name] = []
        fold_scores = {m: [] for m in scoring} 
        info[model_name] = {}


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

                if feature_selection == "rfecv":
                    mask, selector= run_rfecv_once(
                        Xtr, ytr,
                        tuned_estimator,
                        scoring[primary_metric],
                        inner_cv,
                        batches=None
                    )
   
                    selected = X.columns[mask].tolist()
                    selected_count = int(mask.sum())
                    print(f"[RFECV] Selected {selected_count} features")

                    Xtr = Xtr.loc[:, mask]
                    Xte = Xte.loc[:, mask]

                elif feature_selection == "linear_model":
                    mask, selected, fs_info = (
                        run_linear_models(
                            Xtr,
                            ytr,
                            inner_cv=inner_cv,
                            formula=config.linear_formula,
                            filter_to=config.linear_filter_to,
                            alpha=config.linear_alpha,
                            pval_col=config.linear_pval_col,
                            coef_threshold=config.linear_coef_threshold,
                            keep_k=config.linear_keep_k,
                            keep_frac=config.linear_keep_frac,
                            max_features=config.linear_max_features,
                        )
                    )
                    Xtr = Xtr.loc[:, mask]
                    Xte = Xte.loc[:, mask]

                    # Store proteins selected in each fold
                    for inner_fold, proteins in fs_info.items():
                        key = f"outer_{fold_idx+1}/{inner_fold}"
                        info[model_name][key] = set(proteins)
                    info[model_name][f"outer_{fold_idx+1}"] = set(selected)

                else:
                    selected = list(X.columns)
                    mask = np.ones(X.shape[1], dtype=bool)
                    selector = None
                    selected = list(X.columns)
                    selected_count = len(selected)
                    print(f"[NO SELECTION] Using all {selected_count} features")

                tuned_estimator = clone(tuned_estimator).fit(Xtr, ytr)
                y_pred = tuned_estimator.predict(Xte)
                pos_class = 1
                pos_idx = list(tuned_estimator.classes_).index(pos_class)   
                y_pred_prob = tuned_estimator.predict_proba(Xte)[:,pos_idx]

                fold = dict(
                    outer_fold=fold_idx,
                    best_params=best_params,
                    selected_features=selected,
                    selected_feature_count=int(mask.sum()) if feature_selection != "none" else len(selected),
                    confusion_matrix=confusion_matrix(yte, y_pred).tolist(),
                    y_true= yte.tolist(),
                    y_pred=y_pred.tolist(),
                    y_proba=y_pred_prob.tolist(),
                    index= Xte.index.tolist(),
                    classes=tuned_estimator.classes_.tolist()

                )

                classes = tuned_estimator.classes_
                roc_payload = compute_roc(tuned_estimator, Xte, yte,
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
            print(
                f"\n---[DEPLOYMENT MODEL | HP-SEARCH] "
                f"Best params (CV refit on full data): {final_params} ---"
            )

            full_mask = np.ones(X.shape[1], dtype=bool)
            selected_full = list(X.columns)
            full_selector = None
            tol_info = None

            # ---- RFECV (optional) ----
            if feature_selection == "rfecv":
                full_mask, full_selector = run_rfecv_once(
                    X, y,
                    tuned_full,
                    scoring[primary_metric],
                    inner_cv,
                    batches=None,
                )

                # RFECV supported + tolerance enabled
                if config.feature_score_tolerance is not None and full_selector is not None:
                    full_mask, tol_info = select_mask_within_tolerance(
                        full_selector,
                        config.feature_score_tolerance
                    )
                    print(
                        f"[RFECV-TOL] Best={tol_info['best_score']:.4f} | "
                        f"Chosen={tol_info['chosen_score']:.4f} | "
                        f"\n---[FINAL RFECV] {tol_info['chosen_features']}"
                    )

                selected_full = X.columns[full_mask].tolist()

            elif feature_selection == "linear_model":
                full_mask, selected_full, fs_info = (
                    run_linear_models(
                        X,
                        y,
                        inner_cv=None,
                        formula=config.linear_formula,
                        filter_to=config.linear_filter_to,
                        alpha=config.linear_alpha,
                        pval_col=config.linear_pval_col,
                        coef_threshold=config.linear_coef_threshold,
                        keep_k=None,
                        keep_frac=None,
                        max_features=config.linear_max_features,
                        mode="deployment",
                    )
                )
                for inner_fold, proteins in fs_info.items():
                    info[model_name][f"final/{inner_fold}"] = set(proteins)
                info[model_name]["final"] = set(selected_full)
                print(
                    f"\n---[DEPLOYMENT | Linear FS] "
                    f"{len(selected_full)} features selected on full dataset ---"
                )


            else:
                print(f"\n---[NO SELECTION] Using all {len(selected_full)} features")

            # Plot RFECV results (if feature selection was performed)
            #supports_rfecv = hasattr(tuned_full, "coef_") or hasattr(tuned_full, "feature_importances_")

            if feature_selection == "rfecv" and full_selector is not None:
                plot_rfecv_curve(
                    rfecv=full_selector,
                    total_features=X.shape[1],
                    model_name=model_name,
                    output_dir=figures_dir,
                    tolerance=config.feature_score_tolerance,
                    metric_name=primary_metric,
                )

            fit_columns = X.columns[full_mask].tolist()
            final_model = clone(tuned_full).fit(X.loc[:, fit_columns], y)
            final_model.label_encoder_ = le
            final_models[model_name] = (final_model, full_mask)

            # Build the package to save
            package = {
                "model": final_model,
                "mask": full_mask,
                "feature_names": fit_columns,
                "label_encoder": le,
                "best_params": final_params
            }
            final_models[model_name] = package

            # Save to disk (ONE file per model)
            model_dir = Path(config.output_dir) / "saved_models"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(package, model_dir / f"{model_name}.joblib")
            print(f"[SAVED] Final model saved to: {model_dir / f'{model_name}.joblib'}")

            summary = {
                "model": model_name,
                "primary_metric": primary_metric,
                "best_params_full_fit": final_params,
                "selected_feature_count": int(full_mask.sum()) if feature_selection != "none" else len(selected_full),
            }

            for m, values in fold_scores.items():
                summary[f"mean_{m}"] = float(np.mean(values))
                summary[f"std_{m}"] = float(np.std(values)) if len(values) > 1 else 0.0

            results.append(summary)

            if feature_selection == "linear_model":
                save_feature_selection_json(info, output_dir=config.output_dir)

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

