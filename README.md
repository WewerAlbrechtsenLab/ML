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
1. Edit `config/pipeline.yaml` to point to the dataset you want to use and adjust any model or feature-selection options. The training pipeline reads all settings from this file.
2. Run the notebook `notebooks/ML.ipynb` top to bottom. It loads the config, executes nested cross-validation, and logs results automatically. You have to edit the preprocessor if you want to preprocess the data within each split.
Additional data and feature engineering may be necessary and study depended and can be applied before the modeling. 
