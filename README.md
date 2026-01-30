# ML
End-to-end machine learning pipeline using proteomics data. Reproducible workflow with modular code.

## Environment Setup
Use Python 3.11 (or later that remains compatible with the dependencies). Install the required packages with:
```
pip install  git+https://github.com/WewerAlbrechtsenLab/ML.git
```
If you prefer Conda, create a fresh environment first and then run the same pip command inside it:
```
conda create --name ml python=3.11
conda activate ml
%pip install  git+https://github.com/WewerAlbrechtsenLab/ML.git
```

## How to Run
1. Edit `config/pipeline.yaml`.
2. Use the notebook `notebooks/ML.ipynb` as guide. It loads the config, executes nested cross-validation, and logs results automatically.
Additional data and feature engineering may be necessary and study depended and can be applied before the modeling. 


### Cross-validation strategy
- Nested cross-validation is defined by:
  - `outer_splits`: number of outer folds used to estimate generalization performance
  - `inner_splits`: number of inner folds used for hyperparameter tuning and feature selection
  - `random_state`: ensures reproducible data splits

---

### Per–outer-fold workflow
For each model defined in `model_registry` and for each outer fold:

- **Hyperparameter tuning**
  - Hyperparameters are optimized using `RandomizedSearchCV` based on the corresponding entries in `search_spaces`.
  - Optimization is performed on the inner training splits only (`inner_splits`) and scored using the primary evaluation metric.

- **Feature selection (`feature_selection`)**
  - If `feature_selection: rfecv`:
    - Recursive feature elimination with cross-validation (RFECV) is applied on the inner training data.
    - If `feature_score_tolerance > 0`, the smallest feature subset whose CV score is within the specified tolerance of the best score is retained.
  - If `feature_selection: linear_model`:
    - Confounder-adjusted linear models are fit according to `linear_formula`.
    - Features are filtered based on `linear_filter_to`, `linear_alpha`, `linear_pval_col`, and `linear_coef_threshold`.
    - Stability across inner folds is enforced using `linear_keep_k` or `linear_keep_frac`.
    - An optional cap can be applied using `linear_max_features`.
  - If `feature_selection: none`, all features are retained.

- **Model refitting and evaluation**
  - The tuned estimator is refit on the outer-fold training data restricted to the selected features.
  - Predictions are generated on the outer-fold test data.
  - Fold-level outputs include performance metrics, confusion matrices, predicted probabilities, and ROC payloads (binary or multiclass, depending on `task_type`).

---

### Final model (deployment)
After completing all outer folds, a final deployment model is trained using the full dataset (`data_path`):

- **Hyperparameter optimization**
  - Hyperparameter search is re-run on the full dataset using the same `search_spaces`.

- **Feature selection on full data**
  - Feature selection is re-applied using the method specified by `feature_selection`.
  - For RFECV, tolerance-based selection (`feature_score_tolerance`) is applied if enabled, and the RFECV performance curve is saved to `output_dir/figures/`.
  - For linear_model, re-fits the confounder-adjusted linear models once on the full dataset to keep everything that passes the p‑value/coef filters.

- **Model finalization**
  - The final estimator is trained on the full dataset restricted to the selected features.
  - The trained model, feature mask, feature names, label encoder (depending on `task_type`), and optimal hyperparameters are saved to  
    `output_dir/saved_models/{model}.joblib`.

---

### Key configuration-driven behavior
- Input data is loaded from `data_path`
- Output artifacts are written to `output_dir`
- Models and fixed parameters are defined in `model_registry`
- Tunable hyperparameters are defined in `search_spaces`
- Feature selection behavior is controlled entirely by the `feature_selection` section
   