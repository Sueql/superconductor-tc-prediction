# Superconductor Tc Prediction (Python adaptation)

This project is a Python adaptation of the original `main_script_production_9.R` workflow for superconducting critical temperature prediction.  
It is designed around the two CSV files you already have:

- `dataset/train.csv`
- `dataset/unique_m.csv`

## Important note

The original R script does **three** conceptually different things:

1. **Raw-data cleaning** from the original NIMS material database.
2. **Feature engineering** from chemical formula using elemental properties.
3. **Analysis, model training, tuning, and prediction**.

Because this Python project starts from the two CSV files you already own (`train.csv` and `unique_m.csv`), the raw NIMS cleanup stage is **not rerun from scratch**.  
Instead, this adaptation fully implements the downstream functionality on top of your prepared datasets:

- EDA and summary analysis
- univariate plots
- correlation + PCA
- linear baseline + repeated holdout CV
- random forest tuning + repeated holdout CV
- final random forest model + variable importance
- recursive feature elimination (optional)
- optional GBM-style exploration
- final prediction interface

## Project structure

```text
superconductor_tc_prediction/
├── dataset/
│   ├── train.csv
│   ├── unique_m.csv
│   └── README.md
├── config.py
├── formula_parser.py
├── data_loader.py
├── analysis.py
├── models.py
├── training.py
├── predictor.py
├── ui_streamlit.py
├── main.py
├── requirements.txt
├── models/
└── outputs/
```

## What each file does

- `config.py`: paths, constants, feature names, model paths
- `formula_parser.py`: parse chemical formulas like `Ba0.2La1.8Cu1O4`
- `data_loader.py`: load and validate `train.csv` / `unique_m.csv`, add iron/cuprate indicators
- `analysis.py`: plots and statistical summaries corresponding to the EDA sections of the R script
- `models.py`: linear/ridge baseline, random forest tuning/final model, optional GBM, recursive elimination, formula-model training
- `training.py`: end-to-end orchestration
- `predictor.py`: inference utilities for formula and feature-row prediction
- `ui_streamlit.py`: interactive interface
- `main.py`: command-line entry point

## Expected dataset format

### 1) `dataset/train.csv`
Must contain the 81 engineered features plus `critical_temp`.

### 2) `dataset/unique_m.csv`
Must contain:
- 86 elemental-count columns (`H` ... `Rn`)
- `critical_temp`
- `material`

The project assumes the rows in `train.csv` and `unique_m.csv` are aligned row-by-row.

## Environment setup

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Recommended run order

### 1. Check the datasets
```bash
python main.py check-data
```

### 2. Run all EDA / analysis
```bash
python main.py analyze
```

### 3. Train linear baselines
```bash
python main.py train-baselines
```

### 4. Tune random forest
```bash
python main.py tune-rf
```

### 5. Train final RF models (feature model + formula model)
```bash
python main.py train-rf
```

### 6. Optional: recursive feature elimination
```bash
python main.py rfe --max-steps 25
```

### 7. Optional: GBM exploratory run
```bash
python main.py gbm
```

### 8. One-shot full pipeline
```bash
python main.py train-all --with-rfe --rfe-steps 25
```

## Prediction from command line

### Predict from a formula
```bash
python main.py predict-formula --formula "Ba0.2La1.8Cu1O4"
```

### Predict from one CSV row of 81 features
```bash
python main.py predict-feature-row --csv path/to/one_row.csv
```

## Launch the interactive interface

```bash
streamlit run ui_streamlit.py
```

The interface supports:

1. **Formula-based prediction**  
   This uses a random forest trained on the `unique_m.csv` elemental-count representation.  
   It also returns exact/near-exact matches and top similar materials.

2. **81-feature-row prediction**  
   This uses the final random forest trained on `train.csv`.

## Outputs generated

The project writes results to:

- `outputs/analysis/`
- `outputs/linear_full_fit/`
- `outputs/linear_cv/`
- `outputs/rf_tuning/`
- `outputs/rf_final/`
- `outputs/formula_model/`
- `outputs/rfe/` (optional)
- `outputs/gbm_optional/` (optional)

and models to:

- `models/linear_model.joblib`
- `models/ridge_model.joblib`
- `models/rf_feature_model.joblib`
- `models/rf_formula_model.joblib`

## Design choice for the UI

The original R prediction function computes the 81 engineered features directly from the chemical formula via elemental property tables.
Since this Python project is constrained to the two CSV files you already have, the formula-based UI uses `unique_m.csv` directly and trains a separate composition/count-vector random forest.
This keeps the interface fully functional without requiring extra elemental-property data files.
