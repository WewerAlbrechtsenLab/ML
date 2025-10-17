from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import yaml
from scipy import stats as scipy_stats

_SCIPY_DIST_PATTERN = re.compile(r"^scipy\.stats\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$")


@dataclass
class PipelineConfig:
    data_path: str
    target_cols: List[str]
    task_type:  Literal["binary", "multiclass"]
    outer_splits: int
    inner_splits: int
    random_state: int
    feature_selection: Literal["none", "univariate", "rfe", "rfecv"]
    output_dir: str
    feature_cols: Optional[List[str]] = None
    holdout_fraction: Optional[float] = None
    holdout_groups: Optional[List[str]] = None
    model_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    search_spaces: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        raise ValueError(f"Configuration file '{path}' is empty.")

    search_spaces = payload.get("search_spaces")
    if search_spaces:
        payload["search_spaces"] = _coerce_search_spaces(search_spaces)
    return PipelineConfig(**payload)


def _coerce_search_spaces(search_spaces: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    coerced: Dict[str, Dict[str, Any]] = {}
    for model_name, params in search_spaces.items():
        if not isinstance(params, dict):
            coerced[model_name] = params
            continue
        coerced_params: Dict[str, Any] = {}
        for key, value in params.items():
            coerced_params[key] = _coerce_search_value(value)
        coerced[model_name] = coerced_params
    return coerced


def _coerce_search_value(value: Any) -> Any:
    if isinstance(value, str):
        parsed = _parse_scipy_distribution(value)
        return parsed if parsed is not None else value
    if isinstance(value, list):
        return [_coerce_search_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _coerce_search_value(v) for k, v in value.items()}
    return value


def _parse_scipy_distribution(expr: str) -> Any:
    match = _SCIPY_DIST_PATTERN.match(expr.strip())
    if not match:
        return None
    name = match.group("name")
    args_text = match.group("args").strip()
    args, kwargs = _parse_call_arguments(args_text)
    dist_factory = getattr(scipy_stats, name, None)
    if dist_factory is None:
        raise ValueError(f"Unsupported scipy.stats distribution '{name}' in search space.")
    return dist_factory(*args, **kwargs)


def _parse_call_arguments(args_text: str) -> tuple[list[Any], dict[str, Any]]:
    if not args_text:
        return [], {}
    fake_src = f"f({args_text})"
    module = ast.parse(fake_src, mode="exec")
    call = module.body[0].value  # type: ignore[assignment]
    if not isinstance(call, ast.Call):
        raise ValueError(f"Invalid distribution arguments: '{args_text}'")
    args = [_literal_ast_eval(node) for node in call.args]
    kwargs = {
        kw.arg: _literal_ast_eval(kw.value)  # type: ignore[arg-type]
        for kw in call.keywords
        if kw.arg is not None
    }
    return args, kwargs


def _literal_ast_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise ValueError("Search space distribution arguments must be literals.") from exc
