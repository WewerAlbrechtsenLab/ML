from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def _ensure_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _ensure_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Evaluation expects a single target column")
        return y.iloc[:, 0]
    return pd.Series(y)


def evaluate_model_on_dataset(
    model,
    X,
    y,
    positive_label: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute MCC and ROC-AUC for a fitted classifier on a dataset."""

    X_frame = _ensure_frame(X)
    y_series = _ensure_series(y)

    label_encoder = getattr(model, "label_encoder_", None)
    if label_encoder is not None:
        y_true = label_encoder.transform(y_series)
    else:
        y_true = y_series.to_numpy()

    y_pred = model.predict(X_frame)

    metrics: Dict[str, Any] = {
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    y_score = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_frame)
        if proba.ndim == 2 and proba.shape[1] > 1:
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                estimator = model.named_steps.get("estimator")
                classes = getattr(estimator, "classes_", None)
            pos_index = -1
            if positive_label is not None and classes is not None:
                target_label = positive_label
                if label_encoder is not None and isinstance(positive_label, str):
                    target_label = label_encoder.transform([positive_label])[0]
                class_list = list(classes)
                if target_label in class_list:
                    pos_index = class_list.index(target_label)
            y_score = proba[:, pos_index]
        else:
            y_score = proba.ravel()

    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["y_score"] = y_score
    else:
        metrics["roc_auc"] = None

    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred

    return metrics
