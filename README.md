# Superconductor Critical Temperature Prediction

This project predicts superconducting critical temperature, `Tc`, from chemical composition and engineered material descriptors.
It contains a complete modeling pipeline, saved analysis outputs,
command-line prediction tools, and a Streamlit web app for interactive demonstration.

Live app: [https://superconductor-tc-prediction.streamlit.app](https://superconductor-tc-prediction.streamlit.app)

## Highlights

- Uses 21,263 superconducting material records.
- Supports prediction from chemical formulas and engineered feature rows.
- Includes exploratory analysis, baseline models, random forest feature selection, and XGBoost evaluation.
- Uses a selected top-24 feature set for the deployed XGBoost app.
- Provides both local model inference and web-based Streamlit prediction.

## Project Structure

```text
superconductor_tc_prediction/
|-- dataset/
|   |-- train.csv
|   `-- unique_m.csv
|-- models/
|   |-- linear_model.joblib
|   |-- ridge_model.joblib
|   |-- rf_feature_model.joblib
|   |-- rf_feature_model_metadata.json
|   `-- rf_formula_model.joblib
|-- outputs/
|   |-- analysis/
|   |-- rf_final/
|   |-- rfe/
|   |-- rfe_topn_selection/
|   |-- xgb_top24_holdout_test/
|   `-- xgboost/
|-- xgb_tc_app/
|   |-- Home.py
|   |-- model_utils.py
|   |-- feature_config.py
|   |-- formula_parser.py
|   |-- dataset/
|   |-- pages/
|   `-- test/
|-- analysis.py
|-- config.py
|-- data_loader.py
|-- formula_parser.py
|-- main.py
|-- models.py
|-- predictor.py
|-- requirements.txt
|-- training.py
|-- ui_streamlit.py
|-- xgboost.py
|-- rf_top24_test_split.py
`-- xgb_top24_test_split.py
```

Key files:

- `main.py`: command-line entry point for data checks, analysis, training, and prediction.
- `training.py`: end-to-end training pipeline.
- `models.py`: model training, tuning, evaluation, and feature selection.
- `predictor.py`: local inference utilities.
- `xgboost.py`: random forest and XGBoost comparison workflow.
- `ui_streamlit.py`: local random forest Streamlit interface.
- `xgb_tc_app/`: deployable XGBoost Streamlit app.

## Data

The project uses two aligned CSV datasets.

`dataset/train.csv`

- Shape: `21263 x 82`
- Contains 81 engineered material descriptors.
- Uses `critical_temp` as the regression target.

`dataset/unique_m.csv`

- Shape: `21263 x 88`
- Contains element-count columns from `H` to `Rn`.
- Also contains `critical_temp` and `material`.

The row order of the two files is expected to match.
During loading, `data_loader.py` adds `iron` and `cuprate` indicator columns for analysis.

## Setup

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The root `requirements.txt` covers the packages used across the project:
`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `streamlit`, and `xgboost`.

## Main Workflow

Run all commands from the repository root.

Check dataset validity:

```bash
python main.py check-data
```

Expected output for the current files:

```text
train.csv: (21263, 82)
unique_m.csv: (21263, 90)
train_with_indicators: (21263, 85)
```

`unique_m.csv` appears as 90 columns after loading because two indicator columns are added in memory.

Run exploratory analysis:

```bash
python main.py analyze
```

Output directory: `outputs/analysis/`

Train linear and ridge baselines:

```bash
python main.py train-baselines
```

Main outputs:

- `outputs/linear_full_fit/`
- `outputs/linear_cv/`
- `models/linear_model.joblib`
- `models/ridge_model.joblib`

Tune random forest:

```bash
python main.py tune-rf
```

Output directory: `outputs/rf_tuning/`

Train final random forest models:

```bash
python main.py train-rf
```

Main outputs:

- `outputs/rfe/`
- `outputs/rfe_topn_selection/`
- `outputs/rf_final/`
- `outputs/formula_model/`
- `models/rf_feature_model.joblib`
- `models/rf_formula_model.joblib`
- `models/rf_feature_model_metadata.json`

Run the complete pipeline:

```bash
python main.py train-all
```

Optional GBM experiments:

```bash
python main.py train-all --with-gbm
```

## Prediction

Predict from a chemical formula:

```bash
python main.py predict-formula --formula "Ba0.2La1.8Cu1O4"
```

Predict from one row of engineered descriptors:

```bash
python main.py predict-feature-row --csv path/to/one_row.csv
```

The feature-row CSV must contain exactly one row and the required engineered feature columns.
The deployed random forest feature subset is read from `models/rf_feature_model_metadata.json`.

## XGBoost Evaluation

The XGBoost workflow uses the selected top-24 features and compares baseline random forest, tuned random forest, and tuned XGBoost models.

Run the comparison workflow:

```bash
python xgboost.py
```

Main outputs:

- `outputs/xgboost/validation_model_comparison.csv`
- `outputs/xgboost/test_model_comparison.csv`
- `outputs/xgboost/best_model_test_metrics.csv`
- `outputs/xgboost/best_model_feature_importance.csv`
- `outputs/xgboost/best_model_predicted_vs_true.png`
- `outputs/xgboost/best_model_residuals.png`

Run additional top-24 holdout evaluations:

```bash
python xgb_top24_test_split.py
python rf_top24_test_split.py
```

## Current Model Results

The selected model in `outputs/xgboost/best_model_selection.json` is `Tuned_XGB`.

Baseline_RF:

```text
MAE = 5.1415
RMSE = 9.0348
R2 = 0.9291
ACC within 1 K = 0.3224
ACC within 5 K = 0.6929
ACC within 10 K = 0.8439
```

Tuned_RF:

```text
MAE = 5.1417
RMSE = 9.0370
R2 = 0.9291
ACC within 1 K = 0.3231
ACC within 5 K = 0.6929
ACC within 10 K = 0.8441
```

Tuned_XGB:

```text
MAE = 5.1531
RMSE = 8.8826
R2 = 0.9315
ACC within 1 K = 0.2680
ACC within 5 K = 0.6974
ACC within 10 K = 0.8467
```

The Streamlit deployment reports the Tuned_XGB snapshot:

- test MAE: `5.1531 K`
- test RMSE: `8.8826 K`
- test R2: `0.9315`
- accuracy within 10 K: `0.8467`
- selected feature count: `24`

## Selected Top-24 Features

```text
range_ThermalConductivity
wtd_gmean_Valence
wtd_gmean_ThermalConductivity
std_atomic_mass
wtd_entropy_ThermalConductivity
wtd_mean_Valence
wtd_std_ElectronAffinity
range_atomic_radius
wtd_std_ThermalConductivity
wtd_entropy_FusionHeat
wtd_range_atomic_mass
std_Density
mean_Density
gmean_ElectronAffinity
entropy_Density
wtd_std_Valence
wtd_mean_ThermalConductivity
gmean_Density
wtd_mean_atomic_mass
wtd_entropy_atomic_mass
wtd_range_fie
wtd_gmean_ElectronAffinity
wtd_std_atomic_radius
std_atomic_radius
```

## Streamlit Apps

### XGBoost Web App

This is the app intended for web deployment.

Live app:

[https://superconductor-tc-prediction.streamlit.app](https://superconductor-tc-prediction.streamlit.app)

Run locally:

```bash
pip install -r requirements.txt
cd xgb_tc_app
streamlit run Home.py
```

Pages:

- `Formula to Tc`: predicts `Tc` from a chemical formula and shows nearest known materials.
- `Top24 Features to Tc`: predicts `Tc` from the selected 24 descriptors.
- `Batch Prediction`: accepts formula CSV files or top-24 feature CSV files.
- `Model Insights`: displays model metrics and feature importance.

The app trains XGBoost models from the CSV files in `xgb_tc_app/dataset/`, then caches them with Streamlit.
It does not require the large `.joblib` model files in the root `models/` directory.

Sample upload files:

- `xgb_tc_app/test/Formula.csv`
- `xgb_tc_app/test/Top24 feature.csv`

### Local Random Forest App

```bash
streamlit run ui_streamlit.py
```

This interface uses the saved random forest models in `models/` and supports:

- formula-based prediction
- one-row engineered-feature prediction
- display of selected RF features and saved training metrics

## Deployment

For Streamlit Community Cloud or similar platforms, deploy the repository root and set the entry point to:

```text
xgb_tc_app/Home.py
```

Required deployment files:

- `requirements.txt`
- `xgb_tc_app/Home.py`
- `xgb_tc_app/pages/`
- `xgb_tc_app/model_utils.py`
- `xgb_tc_app/feature_config.py`
- `xgb_tc_app/formula_parser.py`
- `xgb_tc_app/dataset/train.csv`
- `xgb_tc_app/dataset/unique_m.csv`

The first page load can take time because the XGBoost models are trained on startup.
Later requests in the same app session use Streamlit caching.
