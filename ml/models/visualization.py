import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay

# -------------------------------------------------------------------
# GLOBAL SETTINGS FOR EDITABLE TEXT
# -------------------------------------------------------------------
def set_editable_text_defaults(font_family: str = "Arial", font_size: int = 12):
    """
    Set matplotlib defaults so exported SVG/PDF keeps text editable.
    """
    rcParams["font.family"] = font_family
    rcParams["font.size"] = font_size

    # IMPORTANT: keep text as text in SVG instead of paths
    rcParams["svg.fonttype"] = "none"

    # Better editable text in PDF/PS
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42


# -------------------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------------------
def plot_confusion(
    json_path,
    model,
    labels=None,
    normalize=None,
    title=None,
    colors=("#fffafa", "#E79600"),
):
    """
    Plot summed confusion matrix from JSON folds.
    Text remains editable when saved as SVG/PDF.
    """
    set_editable_text_defaults()

    with open(json_path) as f:
        results = json.load(f)

    if model not in results:
        raise ValueError(f"Model '{model}' not found in JSON")

    model_folds = results[model]

    cms = [
        np.asarray(fold["confusion_matrix"], dtype=float)
        for fold in model_folds
        if isinstance(fold, dict) and "confusion_matrix" in fold
    ]

    if not cms:
        raise ValueError(f"No confusion matrices found for model '{model}'")

    cm_sum = np.sum(cms, axis=0)

    if normalize == "true":
        cm_plot = cm_sum / cm_sum.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        cm_plot = cm_sum / cm_sum.sum(axis=0, keepdims=True)
    elif normalize == "all":
        cm_plot = cm_sum / cm_sum.sum()
    else:
        cm_plot = cm_sum.astype(int)

    fmt = ".2f" if normalize else "d"
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_plot,
        display_labels=labels
    )
    disp.plot(cmap=custom_cmap, values_format=fmt, ax=ax, colorbar=False)

    if disp.text_ is not None:
        for text in disp.text_.ravel():
            if text is not None:
                text.set_color("black")

    if title is None:
        title = f"Confusion Matrix — {model}"

    ax.set_title(title)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------
# BUTTERFLY PLOT
# -------------------------------------------------------------------
def butterfly_plot(df, var1, var2, var3, error1, error2, group="model"):
    n = len(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, n * 1.5))
    fig.subplots_adjust(wspace=0, left=0.25)  # left margin for labels

    y = np.arange(n)
    height = 0.5
    labels = df[group].tolist()

    # Left plot (MCC) - reversed x axis
    ax1.barh(y, df[var1], height=height, color="#4472c4",
             xerr=[np.zeros(n), df[error1]], capsize=3,
             error_kw=dict(ecolor="black", elinewidth=1))
    ax1.set_xlim(1, 0)
    ax1.set_xlabel("Matthews Correlation Coefficient")
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.yaxis.set_tick_params(length=0)

    # values inside bars
    for i, val in enumerate(df[var1]):
        ax1.text(val / 2, i, f"{val:.3f}", ha="center", va="center",
                 color="white", fontsize=10)

    # Right plot (AUROC)
    ax2.barh(y, df[var2], height=height, color="#ed7d31",
             xerr=[np.zeros(n), df[error2]], capsize=3,
             error_kw=dict(ecolor="black", elinewidth=1))
    ax2.set_xlabel("AUROC")
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    


    # values inside bars
    for i, val in enumerate(df[var2]):
        ax2.text(val / 2, i, f"{val:.3f}", ha="center", va="center",
                 color="white", fontsize=10)

    # Styling
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False)
        ax.set_facecolor("white")
        ax.grid(False)

    fig.patch.set_facecolor("white")
    plt.rcParams["font.family"] = "Arial"

    return fig
# -------------------------------------------------------------------
# FEATURE STABILITY TABLE
# -------------------------------------------------------------------
def extract_feature_stability(fs_json, model_name):
    """
    Returns a DataFrame with:
      protein | n_outer | n_selected | stability | in_final
    """
    model_dict = fs_json[model_name]

    outer_folds = {
        k: set(v)
        for k, v in model_dict.items()
        if k.startswith("outer_") and "/" not in k
    }

    n_outer = len(outer_folds)
    final_set = set(model_dict.get("final", []))
    cv_proteins = set().union(*outer_folds.values()) if outer_folds else set()

    rows = []

    for protein in cv_proteins:
        n_selected = sum(protein in fold for fold in outer_folds.values())
        rows.append({
            "protein": protein,
            "n_outer": n_outer,
            "n_selected": n_selected,
            "stability": n_selected / n_outer if n_outer > 0 else 0.0,
            "in_final": protein in final_set,
        })

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


# -------------------------------------------------------------------
# FEATURE STABILITY PLOT
# -------------------------------------------------------------------
def plot_feature_stability(
    fs_json,
    model_name,
    proteins="all",      # "all" | list[str] | "final"
    top_k=None,
    min_stability=0.0,
    figsize=(10, 5),
    name_mapping=None,
):
    set_editable_text_defaults()

    df = extract_feature_stability(fs_json, model_name)

    if proteins == "final":
        df = df[df["in_final"]]
    elif isinstance(proteins, list):
        df = df[df["protein"].isin(proteins)]

    df = df[df["stability"] >= min_stability]

    if top_k is not None:
        df = df.head(top_k)

    if df.empty:
        raise ValueError("No proteins left after filtering.")

    df = df.copy()
    if name_mapping is not None:
        df["label"] = df["protein"].map(name_mapping).fillna(df["protein"])
    else:
        df["label"] = df["protein"]

    colors = df["in_final"].map({True: "tab:red", False: "tab:gray"})

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df["label"], df["stability"], color=colors)

    ax.set_ylabel("Outer-fold selection frequency")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Feature stability – {model_name}")
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    return fig


# -------------------------------------------------------------------
# RFECV CURVE 
# -------------------------------------------------------------------

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def plot_rfecv_curve(
    rfecv,
    total_features: int,
    model_name: str,
    output_dir: Path | None = None,
    tolerance: float | None = None,
    metric_name: str = "CV score",
):
    """
    RFECV plot using matplotlib.
    """
    set_editable_text_defaults()

    # Data
    scores = np.asarray(rfecv.cv_results_["mean_test_score"])
    stds = np.asarray(rfecv.cv_results_.get("std_test_score", np.zeros_like(scores)))
    n_features = np.asarray(rfecv.cv_results_["n_features"])

    # Sort
    order = np.argsort(n_features)
    n_features = n_features[order]
    scores = scores[order]
    stds = stds[order]

    # Best
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_nfeat = n_features[best_idx]

    # Tolerance
    tol_idx = None
    tol_score = None
    tol_nfeat = None
    same_point = False

    if tolerance is not None:
        threshold = best_score - tolerance
        eligible = np.where(scores >= threshold)[0]
        tol_idx = eligible[np.argmin(n_features[eligible])]
        tol_score = scores[tol_idx]
        tol_nfeat = n_features[tol_idx]
        same_point = tol_idx == best_idx

    # Plot
    plt.figure(figsize=(10, 6))

    plt.fill_between(
        n_features,
        scores - stds,
        scores + stds,
        alpha=0.2,
        label="±1 std"
    )

    plt.plot(
        n_features,
        scores,
        marker="o",
        linewidth=2,
        label="Mean CV score"
    )

    if same_point:
        plt.scatter(
            best_nfeat,
            best_score,
            color="purple",
            s=80,
            marker="D",
            label=f"Best = Selected ({metric_name}={best_score:.4f}, n={best_nfeat})"
        )
    else:
        plt.scatter(
            best_nfeat,
            best_score,
            color="red",
            s=70,
            label=f"Best ({metric_name}={best_score:.4f}, n={best_nfeat})"
        )

    if tol_idx is not None and tol_idx != best_idx:
        plt.scatter(
            tol_nfeat,
            tol_score,
            color="green",
            s=70,
            label=f"Selected ({metric_name}={tol_score:.4f}, n={tol_nfeat})"
        )

    plt.xlabel("Selected features")
    plt.ylabel(metric_name)
    plt.title(f"RFECV – {model_name}")

    tickvals = [1, int(best_nfeat), int(total_features)]
    if tol_nfeat is not None:
        tickvals.append(int(tol_nfeat))
    tickvals = sorted(set(tickvals))
    plt.xticks(tickvals)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()

    # Save
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_svg = output_dir / f"{model_name}_RFECV.svg"
        plt.savefig(out_svg)  

        print(f"[SAVED] RFECV → {out_svg}")

    return plt.gcf()


# -------------------------------------------------------------------
# LOAD OUTER FOLD PREDICTIONS
# -------------------------------------------------------------------
def load_outer_fold_predictions(
    folds_json_path,
    model_name,
):
    """
    Load outer-fold predictions for a given model and return a DataFrame with:
      - prob
      - y_pred
      - y_true
      - outer_fold
      - sample_id as index
    """
    folds_json_path = Path(folds_json_path)
    fold_history = json.loads(folds_json_path.read_text())

    if model_name not in fold_history:
        raise KeyError(f"Model '{model_name}' not found in fold history")

    dfs = []
    for f in fold_history[model_name]:
        df_tmp = pd.DataFrame({
            "prob": f["y_proba"],
            "y_pred": f["y_pred"],
            "y_true": f["y_true"],
            "outer_fold": f["outer_fold"],
            "sample_id": f["index"],
        })
        dfs.append(df_tmp)

    df = pd.concat(dfs, axis=0).set_index("sample_id")
    return df


# -------------------------------------------------------------------
# ROC CURVES
# -------------------------------------------------------------------
def plot_roc_curves(
    leaderboard_csv,
    folds_json_path,
    models="all",          # "all" | str | list[str]
    figsize=(9, 7),
    n_grid=300,
):
    set_editable_text_defaults()

    if isinstance(leaderboard_csv, (str, Path)):
        leaderboard = pd.read_csv(leaderboard_csv)
    else:
        leaderboard = leaderboard_csv

    with open(folds_json_path, "r") as f:
        fold_history = json.load(f)

    if models == "all":
        model_names = leaderboard["model"].tolist()
    else:
        if isinstance(models, str):
            models = [models]
        model_names = models

    if not model_names:
        raise ValueError("No models selected for ROC plotting")

    fpr_grid = np.linspace(0, 1, n_grid)
    fig, ax = plt.subplots(figsize=figsize)

    for model_name in model_names:
        folds = fold_history.get(model_name)

        if not folds:
            raise ValueError(f"No fold data found for model '{model_name}'")

        tprs = []
        aucs = []

        for fold in folds:
            roc = fold.get("roc_curve")
            if roc is None:
                continue

            fpr = np.asarray(roc["fpr"], dtype=float)
            tpr = np.asarray(roc["tpr"], dtype=float)

            tpr_interp = np.interp(fpr_grid, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

            if "test_roc_auc" in fold:
                aucs.append(float(fold["test_roc_auc"]))

        if not tprs:
            raise ValueError(f"No valid ROC curves for model '{model_name}'")

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        mean_auc = np.mean(aucs) if aucs else np.nan
        std_auc = np.std(aucs) if aucs else np.nan

        ax.plot(
            fpr_grid,
            mean_tpr,
            label=f"{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        )

        ax.fill_between(
            fpr_grid,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            alpha=0.2,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Mean ROC Curve (Outer CV)")
    ax.legend(loc="lower right")
    ax.grid(True)

    fig.tight_layout()
    return fig
