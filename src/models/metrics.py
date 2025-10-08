from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def scoring_map(task_type: str) -> Dict[str, str]:
    if task_type == "binary":
        return {
            "f1_weighted": "f1_weighted",
            "mcc": "matthews_corrcoef",
            "roc_auc": "roc_auc",
        }
    return {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "roc_auc_ovr": "roc_auc_ovr",
    }


def summarize_cv_results(model_name: str, cv_results: Dict[str, np.ndarray]) -> pd.Series:
    summary = {key.replace("test_", ""): float(np.mean(values)) for key, values in cv_results.items() if key.startswith("test_")}
    summary["model"] = model_name
    return pd.Series(summary)
