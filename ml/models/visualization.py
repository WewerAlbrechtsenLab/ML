from __future__ import annotations
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
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
    fig = plt.gcf()

    return fig


def butterfly_plot(df, var1, var2, var3, error1, error2, savepdf=True, group = 'model'):
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=False, shared_yaxes=True, horizontal_spacing=0)
    fig.append_trace(go.Bar(x = df[var1], y = df[group], text = df[var1].round(4), error_x=dict(type='data', array=df[error1], arrayminus=np.zeros(len(df))), error_y=dict(type='data', array=df[error1]),
                        textposition='inside', orientation='h', width=0.7, 
                        showlegend=False, marker_color='#4472c4'), 1, 1) # 1,1 represents row 1 column 1
    fig.append_trace(go.Bar(x = df[var2], y = df[group], text = df[var2].round(4), error_x=dict(type='data', array=df[error2], arrayminus=np.zeros(len(df))), error_y=dict(type='data', array=df[error2]),
                 textposition='inside', orientation='h', width=0.7, 
                 showlegend=False, marker_color='#ed7d31'), 1, 2) # 1,2 represents row 1 column 2
    fig.update_xaxes(title_text="Matthews Correlation Coefficient", row=1, col=1, range=[1,0])
    fig.update_xaxes(title_text="AUROC", row=1, col=2)
    fig.update_layout(width=800, height=700, title_x=0.5,xaxis1={'side': 'top'},xaxis2={'side': 'top'},)
    fig.update_layout(template='plotly_white')
    if savepdf:
        fig.write_image('figures/3b.png')
    return fig

def plot_roc_curves(
    leaderboard,
    models="all",   # "all" | str | list[str]
    external_data=None,           # {"y_true": ..., "y_score": ...} or None
    figsize=(9, 7),
):
    def safe_decode(x):
        if isinstance(x, (list, dict)):
            return x
        return ast.literal_eval(x)

    # ---------------------------
    # Select models
    # ---------------------------
    if models == "all":
        lb = leaderboard
    else:
        if isinstance(models, str):
            models = [models]
        lb = leaderboard[leaderboard["model"].isin(models)]

    if lb.empty:
        raise ValueError("No models selected for ROC plotting")

    grid = np.linspace(0, 1, 300)

    plt.figure(figsize=figsize)

    # ======================================================
    # Iterate models
    # ======================================================
    for _, row in lb.iterrows():
        model_name = row["model"]
        folds = safe_decode(row["fold_details"])

        per_class_tprs = {}
        per_class_aucs = {}

        # ---------------------------
        # Collect ROC curves
        # ---------------------------
        for fold in folds:
            roc = fold.get("roc_curve")
            if roc is None:
                continue

            # ---------- Binary ----------
            if "fpr" in roc:
                fpr = np.array(roc["fpr"])
                tpr = np.array(roc["tpr"])
                cls = "binary"

                per_class_tprs.setdefault(cls, []).append(
                    np.interp(grid, fpr, tpr)
                )
                if fold.get("test_roc_auc") is not None:
                    per_class_aucs.setdefault(cls, []).append(
                        fold["test_roc_auc"]
                    )

            # ---------- Multiclass ----------
            elif "per_class" in roc:
                for c in roc["per_class"]:
                    cls = str(c["class_label"])
                    fpr = np.array(c["fpr"])
                    tpr = np.array(c["tpr"])

                    per_class_tprs.setdefault(cls, []).append(
                        np.interp(grid, fpr, tpr)
                    )
                    if fold.get("test_roc_auc") is not None:
                        per_class_aucs.setdefault(cls, []).append(
                            fold["test_roc_auc"]
                        )

        # ---------------------------
        # Plot per class
        # ---------------------------
        for cls, tprs in per_class_tprs.items():
            tprs = np.array(tprs)
            mean_tpr = tprs.mean(axis=0)
            std_tpr = tprs.std(axis=0)

            aucs = per_class_aucs.get(cls, [])
            mean_auc = np.mean(aucs) if aucs else float("nan")
            std_auc = np.std(aucs) if aucs else float("nan")

            label = (
                f"{model_name}"
                if cls == "binary"
                else f"{model_name} — class {cls}"
            )
            label += f" (AUC={mean_auc:.3f}±{std_auc:.3f})"

            plt.plot(grid, mean_tpr, lw=2, label=label)
            plt.fill_between(
                grid,
                mean_tpr - std_tpr,
                mean_tpr + std_tpr,
                alpha=0.15
            )
    # ======================================================
    # Plot EXTERNAL ROC curves
    # ======================================================
    if external_data is not None:
        y_true = external_data["y_true"]
        y_score = external_data["y_score"]

        fpr_ext, tpr_ext, _ = roc_curve(y_true, y_score)
        auc_ext = auc(fpr_ext, tpr_ext)

        plt.plot(
            fpr_ext,
            tpr_ext,
            linestyle="--",
            lw=2.5,
            label=f"External test (AUC={auc_ext:.3f})"
        )


    # ---------------------------
    # Cosmetics
    # ---------------------------
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.3)
    plt.legend()
    fig = plt.gcf()

    return fig

