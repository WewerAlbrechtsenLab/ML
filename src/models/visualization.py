from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, auc

try:  # Display inline when running inside a notebook.
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None  # type: ignore


def _find_roc_column(columns: Iterable[str]) -> str | None:
    """Return the column name that holds mean ROC metrics, if any."""
    preferred = [
        "mean_roc_auc",
        "mean_roc_auc_ovr",
        "mean_roc_auc_ovo",
        "mean_roc_auc_weighted",
    ]

    available = set(columns)
    for candidate in preferred:
        if candidate in available:
            return candidate

    for column in columns:
        if column.startswith("mean_roc_auc"):
            return column
    return None


def _coerce_fold_details(value: Any) -> List[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence):
        result: List[Mapping[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                result.append(item)
        return result
    return []


def _collect_curves(fold_details: Any) -> List[Dict[str, np.ndarray]]:
    curves: List[Dict[str, np.ndarray]] = []
    for fold in _coerce_fold_details(fold_details):
        curve = fold.get("roc_curve")
        if not isinstance(curve, Mapping):
            continue
        fpr = curve.get("fpr")
        tpr = curve.get("tpr")
        thresholds = curve.get("thresholds")
        if not isinstance(fpr, Sequence) or not isinstance(tpr, Sequence):
            continue
        try:
            fpr_arr = np.asarray(fpr, dtype=float)
            tpr_arr = np.asarray(tpr, dtype=float)
            thr_arr = np.asarray(thresholds, dtype=float) if thresholds is not None else None
        except Exception:
            continue
        if fpr_arr.size == 0 or tpr_arr.size == 0:
            continue
        curves.append({"fpr": fpr_arr, "tpr": tpr_arr, "thresholds": thr_arr})
    return curves


def _collect_confusion_matrices(fold_details: Any) -> List[tuple[np.ndarray, List[Any]]]:
    matrices: List[tuple[np.ndarray, List[Any]]] = []
    for fold in _coerce_fold_details(fold_details):
        matrix = fold.get("confusion_matrix")
        if matrix is None:
            continue
        labels = fold.get("confusion_matrix_labels")
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            continue
        if labels is None:
            label_count = arr.shape[0]
            labels = list(range(label_count))
        matrices.append((arr, list(labels)))
    return matrices


def _aggregate_curves(curves: List[Dict[str, np.ndarray]], fpr_grid: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not curves:
        return None, None
    interpolated: List[np.ndarray] = []
    for curve in curves:
        fpr = np.asarray(curve["fpr"], dtype=float)
        tpr = np.asarray(curve["tpr"], dtype=float)
        if fpr.size == 0 or tpr.size == 0:
            continue
        order = np.argsort(fpr)
        fpr_sorted = fpr[order]
        tpr_sorted = tpr[order]
        interp = np.interp(fpr_grid, fpr_sorted, tpr_sorted, left=0.0, right=1.0)
        interpolated.append(interp)
    if not interpolated:
        return None, None
    stack = np.vstack(interpolated)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0) if stack.shape[0] > 1 else None
    return mean, std


def _iter_leaderboard_records(leaderboard: Any) -> List[Dict[str, Any]]:
    if isinstance(leaderboard, pd.DataFrame):
        return leaderboard.to_dict(orient="records")
    if isinstance(leaderboard, Mapping):
        return [dict(leaderboard)]
    if isinstance(leaderboard, Sequence):
        records: List[Dict[str, Any]] = []
        for item in leaderboard:
            if isinstance(item, Mapping):
                records.append(dict(item))
        return records
    raise TypeError("Unsupported leaderboard input; expected DataFrame or sequence of mappings.")


def _aggregate_confusion_matrix(matrices: List[tuple[np.ndarray, List[Any]]]) -> tuple[np.ndarray | None, List[Any] | None]:
    if not matrices:
        return None, None
    agg_df: pd.DataFrame | None = None
    for arr, labels in matrices:
        labels_list = list(labels)
        df = pd.DataFrame(arr, index=labels_list, columns=labels_list, dtype=float)
        agg_df = df if agg_df is None else agg_df.add(df, fill_value=0)
    if agg_df is None:
        return None, None
    agg_df = agg_df.fillna(0)
    order = list(agg_df.index)
    agg_df = agg_df.reindex(index=order, columns=order, fill_value=0)
    return agg_df.to_numpy(), order


def _normalize_confusion_matrix(matrix: np.ndarray, mode: str | None) -> np.ndarray:
    if mode is None:
        return matrix
    normalized = matrix.astype(float, copy=True)
    if mode == "true":
        denom = normalized.sum(axis=1, keepdims=True)
    elif mode == "pred":
        denom = normalized.sum(axis=0, keepdims=True)
    elif mode == "all":
        denom = np.array([[normalized.sum()]])
    else:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}.")
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(normalized, denom, out=np.zeros_like(normalized), where=denom != 0)
    return normalized


def plot_cv_roc_curves(
    leaderboard: Any,
    output_path: Path | str | None = None,
    *,
    show: bool = True,
    fpr_grid: np.ndarray | None = None,
    default_output_dir: Path | None = None,
    models: Sequence[str] | None = None,
) -> Figure:
    """Plot mean ROC curves across CV folds for each model in the leaderboard.

    Parameters
    ----------
    leaderboard:
        Either the in-memory leaderboard DataFrame (with a ``fold_details`` column)
        or a sequence of dictionaries with the same structure (e.g., loaded from metrics.json).
    output_path:
        Optional destination path for the figure. If omitted, the function will attempt
        to place the plot in ``default_output_dir`` or the run directory attached to the
        DataFrame via ``leaderboard.attrs['run_dir']``.
    show:
        When True (default), display the plot (ideal for notebooks).
    fpr_grid:
        Optional array of FPR points to use for interpolation. Defaults to 101 points in [0, 1].
    default_output_dir:
        Fallback directory used when ``output_path`` is not provided.
    models:
        Optional iterable of model names to include. When omitted, every model present in the
        leaderboard is plotted.

    Returns
    -------
    matplotlib.figure.Figure
        The generated ROC summary figure.
    """
    records = _iter_leaderboard_records(leaderboard)
    if not records:
        raise ValueError("Leaderboard is empty; cannot plot ROC curves.")

    grid = np.linspace(0.0, 1.0, 101) if fpr_grid is None else np.asarray(fpr_grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("fpr_grid must be a 1D array of false-positive rates.")

    allowed_models = set(models) if models is not None else None
    model_curves: List[tuple[str, List[Dict[str, np.ndarray]], Dict[str, Any]]] = []
    for record in records:
        model_name = record.get("model")
        if not model_name:
            continue
        if allowed_models is not None and model_name not in allowed_models:
            continue
        curves = _collect_curves(record.get("fold_details"))
        if curves:
            model_curves.append((model_name, curves, record))

    if not model_curves:
        raise ValueError(
            "No ROC curve data found. Ensure the training pipeline captured fold-level probabilities."
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_curves)))

    for idx, (model_name, curves, record) in enumerate(model_curves):
        mean_tpr, std_tpr = _aggregate_curves(curves, grid)
        if mean_tpr is None:
            continue

        mean_auc = float(auc(grid, mean_tpr))
        label = f"{model_name} (mean AUC={mean_auc:.3f})"

        color = colors[idx]
        display_obj = RocCurveDisplay(fpr=grid, tpr=mean_tpr, roc_auc=mean_auc)
        display_obj.plot(
            ax=ax,
            name=label,
            color=color,
            linewidth=2,
            plot_chance_level=False,
        )

        if std_tpr is not None:
            lower = np.clip(mean_tpr - std_tpr, 0.0, 1.0)
            upper = np.clip(mean_tpr + std_tpr, 0.0, 1.0)
            ax.fill_between(grid, lower, upper, color=color, alpha=0.2)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Cross-Validated ROC Curves (Mean of Outer Folds)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    destination = None
    if output_path is not None:
        destination = Path(output_path)
    else:
        run_dir = None
        if isinstance(leaderboard, pd.DataFrame):
            run_dir = leaderboard.attrs.get("run_dir")
            if run_dir:
                default_output_dir = Path(run_dir)
        if default_output_dir is not None:
            destination = Path(default_output_dir) / "cv_roc_summary.png"
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=300)

    if show:
        if display is not None:
            display(fig)
        else:
            plt.show()

    return fig


def plot_cv_confusion_matrices(
    leaderboard: Any,
    output_path: Path | str | None = None,
    *,
    show: bool = True,
    default_output_dir: Path | None = None,
    models: Sequence[str] | None = None,
    normalize: str | None = None,
) -> Figure:
    """Plot aggregated confusion matrices (across outer CV folds) for selected models.

    Parameters
    ----------
    leaderboard:
        In-memory leaderboard (DataFrame) or metrics payload. Must contain ``fold_details`` with
        confusion matrices recorded during training.
    output_path:
        Optional destination for the figure. Defaults to ``cv_confusion_matrices.png`` within the run directory.
    show:
        Whether to display the figure (default ``True``).
    default_output_dir:
        Fallback directory when ``output_path`` is omitted.
    models:
        Optional iterable of model names to plot. By default all models present in ``leaderboard`` are used.
    normalize:
        Normalization mode passed to the confusion matrices: ``"true"``, ``"pred"``, ``"all"``, or ``None`` (counts).
    """
    records = _iter_leaderboard_records(leaderboard)
    if not records:
        raise ValueError("Leaderboard is empty; cannot plot confusion matrices.")

    allowed_models = set(models) if models is not None else None
    norm_mode = normalize.lower() if isinstance(normalize, str) else None

    matrices: List[tuple[str, np.ndarray, List[Any]]] = []
    for record in records:
        model_name = record.get("model")
        if not model_name:
            continue
        if allowed_models is not None and model_name not in allowed_models:
            continue
        collected = _collect_confusion_matrices(record.get("fold_details"))
        agg_matrix, labels = _aggregate_confusion_matrix(collected)
        if agg_matrix is None or labels is None:
            continue
        matrices.append((model_name, agg_matrix, labels))

    if not matrices:
        raise ValueError(
            "No confusion matrix data found. Ensure the training pipeline captured predictions for each fold."
        )

    n_models = len(matrices)
    ncols = min(3, n_models)
    nrows = math.ceil(n_models / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
    if not isinstance(axes, np.ndarray):
        axes_array = np.array([axes])
    else:
        axes_array = axes.flatten()

    for ax in axes_array[n_models:]:
        ax.remove()

    vmax = None if normalize else max(matrix.max() for _, matrix, _ in matrices)

    for ax, (model_name, matrix, labels) in zip(axes_array, matrices):
        display_matrix = _normalize_confusion_matrix(matrix, norm_mode)
        disp = ConfusionMatrixDisplay(confusion_matrix=display_matrix, display_labels=labels)
        disp.plot(ax=ax, colorbar=False)
        if vmax is not None:
            ax.images[-1].set_clim(0, vmax)
        mode_suffix = f" (normalized: {norm_mode})" if norm_mode else ""
        ax.set_title(f"{model_name}{mode_suffix}")

    fig.tight_layout()

    destination = None
    if output_path is not None:
        destination = Path(output_path)
    else:
        run_dir = None
        if isinstance(leaderboard, pd.DataFrame):
            run_dir = leaderboard.attrs.get("run_dir")
            if run_dir:
                default_output_dir = Path(run_dir)
        if default_output_dir is not None:
            destination = Path(default_output_dir) / "cv_confusion_matrices.png"
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=300)

    if show:
        if display is not None:
            plt.show()
        else:
            display(fig)

    return fig


def plot_cv_roc_curves_from_metrics(
    metrics_path: Path | str,
    output_path: Path | str | None = None,
    *,
    show: bool = True,
    fpr_grid: np.ndarray | None = None,
    models: Sequence[str] | None = None,
) -> Figure:
    """Convenience wrapper that loads ``metrics.json`` and plots CV ROC curves.

    Parameters mirror :func:`plot_cv_roc_curves`; the ``models`` argument lets you focus
    on a subset of estimators when the run logged multiple candidates.
    """
    metrics_file = Path(metrics_path)
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    records = payload.get("all_models", [])
    if not records:
        raise ValueError("metrics.json does not contain any model entries.")
    default_dir = metrics_file.parent
    return plot_cv_roc_curves(
        records,
        output_path=output_path,
        show=False,
        fpr_grid=fpr_grid,
        default_output_dir=default_dir,
        models=models,
    )


def plot_cv_confusion_matrices_from_metrics(
    metrics_path: Path | str,
    output_path: Path | str | None = None,
    *,
    show: bool = True,
    models: Sequence[str] | None = None,
    normalize: str | None = None,
) -> Figure:
    """Convenience wrapper that loads ``metrics.json`` and plots confusion matrices."""
    metrics_file = Path(metrics_path)
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    records = payload.get("all_models", [])
    if not records:
        raise ValueError("metrics.json does not contain any model entries.")
    default_dir = metrics_file.parent
    return plot_cv_confusion_matrices(
        records,
        output_path=output_path,
        show=False,
        default_output_dir=default_dir,
        models=models,
        normalize=normalize,
    )
