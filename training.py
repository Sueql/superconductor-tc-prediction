from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis import run_all_analyses
from config import METRICS_JSON, MODEL_DIR, OUTPUT_DIR
from data_loader import load_aligned_datasets
from models import (
    fit_final_random_forest,
    run_linear_baseline_cv,
    train_formula_random_forest,
    train_full_linear_models,
    tune_random_forest,
    repeated_holdout_rf,
    recursive_feature_elimination_rf,
    run_optional_gbm_grid,
)


def run_full_pipeline(run_optional_gbm: bool = False, run_rfe: bool = False, rfe_max_steps: int | None = 25) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df, unique_df, train_with_indicators = load_aligned_datasets()

    run_all_analyses(train_df, unique_df, train_with_indicators, OUTPUT_DIR / 'analysis')

    linear_full = train_full_linear_models(train_df, OUTPUT_DIR / 'linear_full_fit')
    linear_cv = run_linear_baseline_cv(train_df, OUTPUT_DIR / 'linear_cv')

    rf_tuning = tune_random_forest(train_df, OUTPUT_DIR / 'rf_tuning')
    best = rf_tuning.iloc[0].to_dict()
    best_params = {
        'max_features': int(best['max_features']),
        'n_estimators': int(best['n_estimators']),
        'min_samples_leaf': int(best['min_samples_leaf']),
    }

    rf_cv = repeated_holdout_rf(train_df, best_params)
    rf_cv.to_csv(OUTPUT_DIR / 'rf_cv_results.csv', index=False)
    rf_cv.groupby(lambda _: 'RandomForest')[['rmse', 'r2']].describe().to_csv(OUTPUT_DIR / 'rf_cv_summary.csv')

    _, rf_final_metrics = fit_final_random_forest(train_df, OUTPUT_DIR / 'rf_final', best_params)
    train_formula_random_forest(unique_df, OUTPUT_DIR / 'formula_model', best_params)

    gbm_results_path = None
    if run_optional_gbm:
        gbm_results = run_optional_gbm_grid(train_df, OUTPUT_DIR / 'gbm_optional')
        gbm_results_path = str((OUTPUT_DIR / 'gbm_optional' / 'gbm_grid_results.csv').resolve())

    rfe_results_path = None
    if run_rfe:
        rfe_df = recursive_feature_elimination_rf(train_df, OUTPUT_DIR / 'rfe', best_params, max_steps=rfe_max_steps)
        rfe_results_path = str((OUTPUT_DIR / 'rfe' / 'rfe_variable_importance.csv').resolve())

    summary = {
        'linear_full_fit': {k: {'rmse': v.rmse, 'r2': v.r2} for k, v in linear_full.items()},
        'linear_cv_mean': linear_cv.groupby('model')[['rmse', 'r2']].mean().to_dict(),
        'rf_best_params': best_params,
        'rf_cv_mean': rf_cv[['rmse', 'r2']].mean().to_dict(),
        'rf_final_metrics_csv': str((OUTPUT_DIR / 'rf_final' / 'rf_final_metrics.csv').resolve()),
        'gbm_results_csv': gbm_results_path,
        'rfe_results_csv': rfe_results_path,
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run_fast_train() -> dict:
    return run_full_pipeline(run_optional_gbm=False, run_rfe=False)
