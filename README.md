# ML
End-to-end machine learning pipeline using proteomics data. Reproducible workflow with modular code.

## How to Run
Execute the training and evaluation pipeline by running the jupyter notebooks/ML.ipynb
The run will be logged.

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
