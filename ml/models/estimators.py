from __future__ import annotations

from importlib import import_module
from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.utils.config import PipelineConfig


def _import_from_path(path: str) -> type[BaseEstimator]:
    module_path, class_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def inspect_hyperparameters(models: dict, config: PipelineConfig):
    """
    Prints all models and their hyperparameter search spaces
    Helps catch typos, wrong parameter names, or empty grids.
    """
    search_spaces = getattr(config, "search_spaces", {})

    print("\n===== MODEL & HYPERPARAMETER CHECK =====\n")

    for model_name, estimator in models.items():
        print(f"• Model: {model_name}")
        print(f"  Estimator: {estimator.__class__.__name__}")

        grid = search_spaces.get(model_name, {})

        if not grid:
            print("  Hyperparameters: <none> (using base estimator)")
        else:
            print("  Hyperparameters:")
            for param, values in grid.items():
                print(f"    - {param}: {values}")

        # quick validation warning — avoid runtime crashes
        invalid_params = [
            p for p in grid.keys() if not hasattr(estimator, "get_params") 
            or p not in estimator.get_params()
        ]

        if invalid_params:
            print(f"  WARNING: Invalid hyperparameters for {model_name}: {invalid_params}")

        print()

    print("===== END CHECK =====\n")


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
