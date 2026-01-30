from __future__ import annotations
import ast
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.feature_selection import RFECV
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json

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

def extract_feature_stability(fs_json, model_name):
    """
    Returns a DataFrame with:
      protein | n_outer | n_selected | stability | in_final
    """

    model_dict = fs_json[model_name]

    # identify outer folds
    outer_folds = {
        k: set(v)
        for k, v in model_dict.items()
        if k.startswith("outer_") and "/" not in k
    }

    n_outer = len(outer_folds)

    # final model proteins
    final_set = set(model_dict.get("final", []))

    # proteins selected at least once during CV
    cv_proteins = set().union(*outer_folds.values()) if outer_folds else set()

    rows = []

    # ---- CV-selected proteins ----
    for protein in cv_proteins:
        n_selected = sum(
            protein in fold for fold in outer_folds.values()
        )

        rows.append({
            "protein": protein,
            "n_outer": n_outer,
            "n_selected": n_selected,
            "stability": n_selected / n_outer if n_outer > 0 else 0.0,
            "in_final": protein in final_set,
        })

    # ---- FINAL-ONLY proteins → zero stability ----
    final_only = final_set - cv_proteins

    for protein in final_only:
        rows.append({
            "protein": protein,
            "n_outer": n_outer,
            "n_selected": 0,
            "stability": 0.0,
            "in_final": True,
        })

    df = (
        pd.DataFrame(rows)
        .sort_values(
            ["in_final", "stability", "protein"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    return df


def plot_feature_stability(
    fs_json,
    model_name,
    proteins="all",      # "all" | list[str] | "final"
    top_k=None,
    min_stability=0.0,
    figsize=(10, 5),
    name_mapping=None,
):
    """
    Bar plot of feature stability from outer CV folds.
    """

    df = extract_feature_stability(fs_json, model_name)

    # ---- protein selection ----
    if proteins == "final":
        df = df[df["in_final"]]
    elif isinstance(proteins, list):
        df = df[df["protein"].isin(proteins)]

    df = df[df["stability"] >= min_stability]

    if top_k is not None:
        df = df.head(top_k)

    if df.empty:
        raise ValueError("No proteins left after filtering.")

    # ---- apply gene-name mapping ----
    if name_mapping is not None:
        df["label"] = df["protein"].map(name_mapping).fillna(df["protein"])
    else:
        df["label"] = df["protein"]

    # ---- plotting ----
    colors = df["in_final"].map({True: "tab:red", False: "tab:gray"})

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df["label"], df["stability"], color=colors)

    ax.set_ylabel("Outer-fold selection frequency")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Feature stability – {model_name}")

    ax.axhline(0.5, linestyle="--", linewidth=1)
    #ax.text(0, 0.52, "50% stability", fontsize=9)

    ax.tick_params(axis="x", rotation=90)

    # legend
    from matplotlib.patches import Patch
    # ax.legend(
    #     handles=[
    #         Patch(color="tab:red", label="In final model"),
    #         Patch(color="tab:gray", label="Not in final model"),
    #     ],
    #     frameon=False,
    # )
    plt.tight_layout()
    fig = plt.gcf()

    return fig

def plot_rfecv_curve(
    rfecv: RFECV,
    total_features: int,
    model_name: str,
    output_dir: Path,
    tolerance: float | None = None,
    metric_name: str = "CV score",
):
    """
    Plot RFECV performance curve with optional tolerance-based selection.

    Parameters
    ----------
    rfecv : RFECV
        Fitted RFECV object
    total_features : int
        Total number of features before selection
    model_name : str
        Model name (used for filename)
    output_dir : Path
        Directory where the figure will be saved
    tolerance : float, optional
        Score tolerance for near-optimal feature selection
    metric_name : str
        Label for y-axis
    """

    import numpy as np
    import plotly.graph_objects as go

    scores = rfecv.cv_results_["mean_test_score"]
    stds = rfecv.cv_results_.get("std_test_score", np.zeros_like(scores))
    n_features = rfecv.cv_results_["n_features"]

    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_nfeat = n_features[best_idx]

    tol_idx = None
    if tolerance is not None:
        threshold = best_score - tolerance
        eligible = np.where(scores >= threshold)[0]
        tol_idx = eligible[np.argmin(n_features[eligible])]
        tol_score = scores[tol_idx]
        tol_nfeat = n_features[tol_idx]
        same_point = (tol_idx is not None and tol_idx == best_idx)

    error_max = scores + stds
    error_min = scores - stds

    fig = go.Figure()

    # ±1 std band
    fig.add_trace(go.Scatter(
        x=np.concatenate([n_features, n_features[::-1]]),
        y=np.concatenate([error_max, error_min[::-1]]),
        fill="toself",
        opacity=0.25,
        line=dict(color="lightgray"),
        name="±1 std",
    ))

    # Mean curve
    fig.add_trace(go.Scatter(
        x=n_features,
        y=scores,
        mode="lines+markers",
        line=dict(color="firebrick", width=2),
        name="Mean CV score",
    ))

    # --- Best point ---
    if same_point:
        fig.add_trace(go.Scatter(
            x=[best_nfeat],
            y=[best_score],
            mode='markers',
            marker=dict(color='purple', size=11, symbol='diamond'),
            name=f"Best = Selected (F1={best_score:.4f}, n={best_nfeat})",
        ))

    else:
        # --- Best point ---
        fig.add_trace(go.Scatter(
            x=[best_nfeat],
            y=[best_score],
            mode='markers',
            marker=dict(color='red', size=10),
            name=f"Best (F1={best_score:.4f}, n={best_nfeat})",
        ))

    # Tolerance-selected point
    if tol_idx is not None and tol_idx != best_idx:
        fig.add_trace(go.Scatter(
            x=[tol_nfeat],
            y=[tol_score],
            mode="markers",
            marker=dict(color="green", size=10),
            name=f"Selected (F1={tol_score:.4f}, n={tol_nfeat})",
        ))

    fig.update_layout(
        width=1000,
        height=650,
        margin=dict(l=90, r=40, t=100, b=80),
        title=f"RFECV – {model_name}",
        yaxis_title=metric_name,
        xaxis=dict(
            title=f"Selected features)",
            range=[1, total_features + 1],
            tickmode="array",
                            tickvals=[1, best_nfeat, tol_nfeat, total_features]
                            if tol_idx is not None else [1, best_nfeat, total_features],
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}_RFECV.svg"
    fig.write_image(out_path)

    print(f"[SAVED] RFECV plot → {out_path}")

    return fig

def load_outer_fold_predictions(
    folds_json_path,
    model_name,
):
    """
    Load outer-fold predictions for a given model and return a DataFrame with:
      - prob        : predicted probability (positive class)
      - y_pred      : predicted label
      - y_true      : true label
      - outer_fold  : outer CV fold index
      - index   : sample identifier (as index)

    Parameters
    ----------
    folds_json_path : str or Path
        Path to leaderboard_*_folds.json
    model_name : str
        Model key inside fold_history (e.g. 'logistic_regression')

    Returns
    -------
    pd.DataFrame
    """

    folds_json_path = Path(folds_json_path)
    fold_history = json.loads(folds_json_path.read_text())

    if model_name not in fold_history:
        raise KeyError(f"Model '{model_name}' not found in fold history")

    dfs = []
    for f in fold_history[model_name]:
        df_tmp = pd.DataFrame(
            {
                "prob": f["y_proba"],
                "y_pred": f["y_pred"],
                "y_true": f["y_true"],
                "outer_fold": f["outer_fold"],
            }
        )

        # attach sample index safely
        df_tmp["sample_id"] = f["index"]
        dfs.append(df_tmp)

    df = pd.concat(dfs, axis=0).set_index("sample_id")

    return df
