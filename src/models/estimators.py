from __future__ import annotations

from importlib import import_module
from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.config import PipelineConfig


def _import_from_path(path: str) -> type[BaseEstimator]:
    module_path, class_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def build_default_registry(task_type: str) -> Dict[str, BaseEstimator]:
    estimators: Dict[str, BaseEstimator] = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=42),
    }

    if task_type == "multiclass":
        estimators["logistic_regression"] = LogisticRegression(max_iter=1000, multi_class="auto")

    return estimators


def build_models(config: PipelineConfig) -> Dict[str, BaseEstimator]:
    if not config.model_registry:
        return build_default_registry(config.task_type)

    models: Dict[str, BaseEstimator] = {}
    for name, spec in config.model_registry.items():
        class_path = spec.get("classname")
        params = spec.get("params", {})
        if not class_path:
            raise ValueError(f"Model registry entry '{name}' is missing 'classname'")
        estimator_cls = _import_from_path(class_path)
        models[name] = estimator_cls(**params)
    return models
