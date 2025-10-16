# ML
End-to-end machine learning pipeline using proteomics data. Reproducible workflow with modular code.

## Environment Setup
Use Python 3.9 (or later that remains compatible with the dependencies). Install the required packages with:
```
pip install -r requirements.txt
```
If you prefer Conda, create a fresh environment first and then run the same pip command inside it:
```
conda create --name ml python=3.9
conda activate ml
pip install -r requirements.txt
```

## How to Run
1. Edit `config/pipeline.yaml` to point to the dataset you want to use and adjust any model or feature-selection options. The training pipeline reads all settings from this file.
2. Run the notebook `notebooks/ML.ipynb` top to bottom. It loads the config, executes nested cross-validation, and logs results automatically.
Additional data and feature engineering may be necessary and study depended and can be applied before the modeling. 

## Run Logging
Each training creates a deterministic run directory under `outputs/runs/<run_id>/`.  
Outputs captured per run:
- `config.yaml` – pipeline configuration.
- `git_sha.txt` – commit identifier used for the run.
- `data_hash.txt` – SHA-256 of the raw data file referenced by the config.
- `leaderboard.csv` – leaderboard DataFrame exported from nested CV.
- `metrics.json` – JSON summary of model metrics (best model plus per-model records).
- `best_model.joblib` – serialized best estimator pipeline.
- `run_manifest.json` – metadata including timestamp, user, seed, and library versions.

`run_id` is a short SHA derived from the config, data hash, and git commit. Identical inputs resolve to the same directory so repeated runs with no input changes don’t create duplicates.
