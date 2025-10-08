from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import yaml


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
    return PipelineConfig(**payload)
