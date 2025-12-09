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


def plot_all_roc_from_leaderboard(leaderboard):
    """
    Plots mean ROC curves for all models.
    Supports binary + multiclass ROC.
    Automatically decodes fold_details from CSV.
    """

    # ---- 1) Decode fold_details safely ----
    leaderboard = leaderboard.copy()

    def safe_decode(x):
        if isinstance(x, (dict, list)):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            print("Failed to decode fold_details entry:", x)
            return []
    
    leaderboard["fold_details"] = leaderboard["fold_details"].apply(safe_decode)

    # ---- 2) Prepare plot ----
    plt.figure(figsize=(8, 7))
    grid = np.linspace(0, 1, 300)

    # ---- 3) Iterate models ----
    for _, row in leaderboard.iterrows():
        model = row["model"]
        folds = row["fold_details"]

        per_class_curves = {}

        # ---- Collect ROC curves ----
        for fold in folds:
            roc = fold.get("roc_curve")
            if roc is None:
                continue

            # Binary case
            if "fpr" in roc:
                per_class_curves.setdefault("binary", []).append(
                    (np.array(roc["fpr"]), np.array(roc["tpr"]))
                )

            # Multiclass
            elif "per_class" in roc:
                for c in roc["per_class"]:
                    cls = c["class_label"]
                    fpr = np.array(c["fpr"])
                    tpr = np.array(c["tpr"])
                    per_class_curves.setdefault(cls, []).append((fpr, tpr))

        # ---- Plot averaged ROC curves ----
        for cls, curves in per_class_curves.items():
            tpr_interp = []
            for fpr, tpr in curves:
                tpr_interp.append(np.interp(grid, fpr, tpr))

            tpr_interp = np.array(tpr_interp)
            mean_tpr = tpr_interp.mean(axis=0)
            std_tpr = tpr_interp.std(axis=0)
            auc_value = auc(grid, mean_tpr)
            
            per_fold_auc = [fold["test_roc_auc"] for fold in folds if fold.get("roc_curve") is not None and fold.get("test_roc_auc") is not None
    ]
            mean_test_auc = np.mean(per_fold_auc) if per_fold_auc else float("nan")
            std_test_auc = np.std(per_fold_auc) if per_fold_auc else float("nan")

            label = (
                f"{model} — {cls} "
                f"(avg ROC AUC={auc_value:.3f}, test AUC={mean_test_auc:.3f}±{std_test_auc:.3f})"
            )
            plt.plot(grid, mean_tpr, lw=2, label=label)
            plt.fill_between(grid, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.15)

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves — All Models")
    plt.grid(alpha=0.3)
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
