from __future__ import annotations
from src.utils.config import PipelineConfig 
from typing import Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

def _select_target_column(df: pd.DataFrame, candidates: Optional[Sequence[str]]) -> Optional[str]:
    if not candidates:
        return None
    for name in candidates:
        if name in df.columns:
            return name
    return None


def prepare_train_holdout_split(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return annotated data plus train and hold-out subsets."""

    annotated = df.copy()
    annotated["study_split"] = "train"

    if config.holdout_groups:
        group_mask = annotated["group"].isin(config.holdout_groups)
        annotated.loc[group_mask, "study_split"] = "holdout"

    holdout_mask = annotated["study_split"] == "holdout"

    target_column = _select_target_column(annotated, config.target_cols)

    if config.holdout_fraction and config.holdout_fraction > 0:
        train_pool = annotated.loc[~holdout_mask].copy()
        stratify_values = None
        if target_column is not None:
            target_values = train_pool[target_column]
            if target_values.notna().nunique() > 1:
                stratify_values = target_values
        train_idx, holdout_idx = train_test_split(
            train_pool.index,
            test_size=config.holdout_fraction,
            random_state=config.random_state,
            stratify=stratify_values,
        )
        annotated.loc[holdout_idx, "study_split"] = "holdout"
        holdout_mask = annotated["study_split"] == "holdout"

    train_df = annotated.loc[annotated["study_split"] == "train"].copy()
    holdout_df = annotated.loc[holdout_mask].copy()

    return annotated, train_df, holdout_df
