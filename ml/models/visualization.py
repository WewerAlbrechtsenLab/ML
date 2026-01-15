from __future__ import annotations
import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, auc
from plotly.subplots import make_subplots
import plotly.graph_objects as go

try:  # Display inline when running inside a notebook.
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None  # type: ignore




def safe_decode_fold_details(x):
    # Already decoded
    if isinstance(x, (list, dict)):
        return x
    
    if not isinstance(x, str):
        return []

    # Replace "inf" with a numeric value BEFORE parsing
    cleaned = x.replace("inf", "1e309")  # float('inf') equivalent
    
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        print("FAILED TO DECODE ENTRY:", x[:200], "...")
        return []

def plot_roc_from_leaderboard(
    leaderboard,
    models="all",                 # "all" | str | list[str]
    title=None,                   # optional custom title
    figsize=(8, 7),
    n_grid=300,
    legend=True,
):
    """
    Plot mean ROC curves from a leaderboard with per-fold stored roc_curve info.

    Parameters
    ----------
    leaderboard : pd.DataFrame
        Must contain columns: ["model", "fold_details"].
        fold_details can be a python object or a string representation of it.
    models : "all" | str | list[str]
        - "all": plot every model in leaderboard
        - str: plot only that model name
        - list[str]: plot only those model names
    title : str | None
        Plot title. Defaults to "ROC Curves — <models...>".
    """

    # ---- 1) Decode fold_details safely ----
    lb = leaderboard.copy()

    def safe_decode(x):
        if isinstance(x, (dict, list)):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            print("Failed to decode fold_details entry:", x)
            return []

    lb["fold_details"] = lb["fold_details"].apply(safe_decode)

    # ---- 2) Filter models ----
    if models != "all":
        if isinstance(models, str):
            selected = {models}
        else:
            selected = set(models)
        lb = lb[lb["model"].isin(selected)].copy()

    if lb.empty:
        available = sorted(set(leaderboard["model"].astype(str)))
        raise ValueError(
            f"No rows left after filtering. Requested={models}. "
            f"Available models={available}"
        )

    # ---- 3) Prepare plot ----
    plt.figure(figsize=figsize)
    grid = np.linspace(0, 1, n_grid)

    # ---- 4) Iterate models ----
    for _, row in lb.iterrows():
        model = row["model"]
        folds = row["fold_details"] or []

        per_class_curves = {}

        # ---- Collect ROC curves ----
        for fold in folds:
            roc = (fold or {}).get("roc_curve")
            if not roc:
                continue

            # Binary case
            if "fpr" in roc and "tpr" in roc:
                per_class_curves.setdefault("binary", []).append(
                    (np.array(roc["fpr"]), np.array(roc["tpr"]))
                )

            # Multiclass case
            elif "per_class" in roc:
                for c in roc["per_class"]:
                    cls = c.get("class_label", "unknown")
                    fpr = np.array(c.get("fpr", []))
                    tpr = np.array(c.get("tpr", []))
                    if fpr.size and tpr.size:
                        per_class_curves.setdefault(cls, []).append((fpr, tpr))

        if not per_class_curves:
            continue

        # ---- Plot averaged ROC curves ----
        for cls, curves in per_class_curves.items():
            tpr_interp = []
            for fpr, tpr in curves:
                order = np.argsort(fpr)
                fpr_sorted = fpr[order]
                tpr_sorted = tpr[order]
                tpr_interp.append(np.interp(grid, fpr_sorted, tpr_sorted))

            tpr_interp = np.asarray(tpr_interp)
            mean_tpr = tpr_interp.mean(axis=0)
            std_tpr = tpr_interp.std(axis=0)

            auc_value = auc(grid, mean_tpr)

            per_fold_auc = [
                fold.get("test_roc_auc")
                for fold in folds
                if fold and fold.get("roc_curve") is not None and fold.get("test_roc_auc") is not None
            ]
            mean_test_auc = float(np.mean(per_fold_auc)) if per_fold_auc else float("nan")
            std_test_auc = float(np.std(per_fold_auc)) if per_fold_auc else float("nan")

            label = (
                f"{model} — {cls} "
                f"(avg ROC AUC={auc_value:.3f}, test AUC={mean_test_auc:.3f}±{std_test_auc:.3f})"
            )
            plt.plot(grid, mean_tpr, lw=2, label=label)
            plt.fill_between(grid, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.15)

    # ---- 5) Cosmetics ----
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    if title is None:
        if models == "all":
            title = "ROC Curves — All Models"
        else:
            title = "ROC Curves — Selected Models"
    plt.title(title)

    plt.grid(alpha=0.3)
    if legend:
        plt.legend()
    plt.show()


def plot_confusion_from_leaderboard(leaderboard, model, normalize=None):
    folds = leaderboard.loc[leaderboard.model == model, "fold_details"].iloc[0]

    # collect & sum matrices
    matrices = []
    for fold in folds:
        cm = np.asarray(fold["confusion_matrix"], dtype=float)
        matrices.append(cm)

    agg = np.sum(matrices, axis=0)

    # normalize if needed
    if normalize == "true":
        agg = agg / agg.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        agg = agg / agg.sum(axis=0, keepdims=True)
    elif normalize == "all":
        agg = agg / agg.sum()

    # format correctly
    fmt = ".2f" if normalize else "d"
    if not normalize:
        agg = agg.round().astype(int)

    disp = ConfusionMatrixDisplay(confusion_matrix=agg)
    disp.plot(values_format=fmt, cmap="Blues")
    plt.title(f"Confusion Matrix (outer CV) — {model}")
    plt.show()


def butterfly_plot(df, var1, var2, var3, error1, error2, savepdf=True, group = 'model'):
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=False, shared_yaxes=True, horizontal_spacing=0)
    fig.append_trace(go.Bar(x = df[var1], y = df[group], text = df[var1], error_x=dict(type='data', array=df[error1]), error_y=dict(type='data', array=df[error1]),
                        textposition='inside', orientation='h', width=0.7, 
                        showlegend=False, marker_color='#4472c4'), 1, 1) # 1,1 represents row 1 column 1
    fig.append_trace(go.Bar(x = df[var2], y = df[group], text = df[var2], error_x=dict(type='data', array=df[error2]), error_y=dict(type='data', array=df[error2]),
                 textposition='inside', orientation='h', width=0.7, 
                 showlegend=False, marker_color='#ed7d31'), 1, 2) # 1,2 represents row 1 column 2
    fig.update_xaxes(title_text="Matthews Correlation Coefficient", row=1, col=1, range=[1,0])
    fig.update_xaxes(title_text="AUROC", row=1, col=2)
    fig.update_layout(width=800, height=700, title_x=0.5,xaxis1={'side': 'top'},xaxis2={'side': 'top'},)
    fig.update_layout(template='plotly_white')
    if savepdf:
        fig.write_image('figures/3b.png')
    fig.show()
