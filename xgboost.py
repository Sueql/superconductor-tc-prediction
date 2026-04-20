from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split
from xgboost import XGBRegressor

from config import OUTPUT_DIR, RANDOM_STATE, TARGET_COLUMN
from data_loader import load_aligned_datasets
from progress_utils import log, progress, stage_end, stage_start

TOP24_FEATURES = [
    'range_ThermalConductivity',
    'wtd_gmean_Valence',
    'wtd_gmean_ThermalConductivity',
    'std_atomic_mass',
    'wtd_entropy_ThermalConductivity',
    'wtd_mean_Valence',
    'wtd_std_ElectronAffinity',
    'range_atomic_radius',
    'wtd_std_ThermalConductivity',
    'wtd_entropy_FusionHeat',
    'wtd_range_atomic_mass',
    'std_Density',
    'mean_Density',
    'gmean_ElectronAffinity',
    'entropy_Density',
    'wtd_std_Valence',
    'wtd_mean_ThermalConductivity',
    'gmean_Density',
    'wtd_mean_atomic_mass',
    'wtd_entropy_atomic_mass',
    'wtd_range_fie',
    'wtd_gmean_ElectronAffinity',
    'wtd_std_atomic_radius',
    'std_atomic_radius',
]

OUTDIR = OUTPUT_DIR / 'xgboost'


# ----------------------------
# Metrics helpers
# ----------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def accuracy_within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, tolerances: List[float]) -> Dict[str, float]:
    out = {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(rmse(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
    }
    for t in tolerances:
        out[f'ACC@±{int(t)}K'] = accuracy_within_tolerance(y_true, y_pred, t)
    return out


# ----------------------------
# I/O helpers
# ----------------------------
def save_predictions_csv(
    path: Path,
    materials: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float,
    model_name: str,
    extra_metrics: Dict[str, float],
) -> None:
    pred_ok = np.abs(y_true - y_pred) <= tolerance
    df = pd.DataFrame({
        'material': materials.to_numpy(),
        'true_Tc': y_true,
        'predicted_Tc': y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'prediction_result': np.where(pred_ok, 'Correct', 'Error'),
    })

    summary = {
        'material': f'OVERALL_{model_name.upper()}_TEST_SUMMARY',
        'true_Tc': np.nan,
        'predicted_Tc': np.nan,
        'abs_error': np.nan,
        'prediction_result': '; '.join([
            f'tolerance={tolerance:.2f}K',
            *[f'{k}={v:.6f}' for k, v in extra_metrics.items()],
            f'test_size={len(y_true)}',
        ]),
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    df.to_csv(path, index=False)


# ----------------------------
# Plot helpers
# ----------------------------
def plot_metric_comparison(validation_metrics_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=140)
    axes = axes.flatten()
    cols = ['MAE', 'RMSE', 'R2', 'ACC@±10K']
    titles = [
        'Validation MAE (lower better)',
        'Validation RMSE (lower better)',
        'Validation R² (higher better)',
        'Validation Accuracy within ±10K (higher better)',
    ]
    for ax, col, title in zip(axes, cols, titles):
        ax.bar(validation_metrics_df['model'], validation_metrics_df[col])
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_metric_comparison_full(metrics_df: pd.DataFrame, save_path: Path, prefix: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    axes = axes.flatten()
    cols = ['MAE', 'RMSE', 'R2', 'ACC@±1K', 'ACC@±5K', 'ACC@±10K']
    titles = [
        f'{prefix} MAE', f'{prefix} RMSE', f'{prefix} R²',
        f'{prefix} ACC@±1K', f'{prefix} ACC@±5K', f'{prefix} ACC@±10K',
    ]
    for ax, col, title in zip(axes, cols, titles):
        ax.bar(metrics_df['model'], metrics_df[col])
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=20)
        if col.startswith('ACC') or col == 'R2':
            ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_tolerance_curve(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    tolerances = np.arange(1, 21)
    accs = [accuracy_within_tolerance(y_true, y_pred, float(t)) for t in tolerances]
    plt.figure(figsize=(8, 5), dpi=140)
    plt.plot(tolerances, accs, marker='o')
    plt.xlabel('Tolerance (K)')
    plt.ylabel('Accuracy within tolerance')
    plt.title('Accuracy vs tolerance for best model')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_predicted_vs_true(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str) -> None:
    plt.figure(figsize=(6.5, 6), dpi=140)
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.xlabel('True Tc (K)')
    plt.ylabel('Predicted Tc (K)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_predicted_vs_true_by_model(pred_df: pd.DataFrame, save_path: Path, split_name: str) -> None:
    models = ['Baseline_RF', 'Tuned_RF', 'Tuned_XGB']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=140)
    y_true = pred_df['true_Tc'].to_numpy()
    lo = float(np.min(y_true))
    hi = float(np.max(y_true))
    for ax, model in zip(axes, models):
        y_pred = pred_df[f'pred_{model}'].to_numpy()
        ax.scatter(y_true, y_pred, s=10, alpha=0.6)
        ax.plot([lo, hi], [lo, hi], 'r--')
        ax.set_title(f'{model}\n{split_name}: predicted vs true')
        ax.set_xlabel('True Tc (K)')
        ax.set_ylabel('Predicted Tc (K)')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    axes[0].scatter(y_pred, residuals, s=12, alpha=0.6)
    axes[0].axhline(0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Tc (K)')
    axes[0].set_ylabel('Residual (true - pred)')
    axes[0].set_title('Residual scatter')

    axes[1].hist(residuals, bins=40)
    axes[1].set_xlabel('Residual (K)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual histogram')

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_residual_histograms_by_model(pred_df: pd.DataFrame, save_path: Path, split_name: str) -> None:
    models = ['Baseline_RF', 'Tuned_RF', 'Tuned_XGB']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=140)
    y_true = pred_df['true_Tc'].to_numpy()
    for ax, model in zip(axes, models):
        residuals = y_true - pred_df[f'pred_{model}'].to_numpy()
        ax.hist(residuals, bins=35)
        ax.axvline(0, linestyle='--')
        ax.set_title(f'{model}\n{split_name}: residual histogram')
        ax.set_xlabel('Residual (K)')
        ax.set_ylabel('Count')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_absolute_error_boxplot(pred_df: pd.DataFrame, save_path: Path, split_name: str) -> None:
    models = ['Baseline_RF', 'Tuned_RF', 'Tuned_XGB']
    data = [pred_df[f'abs_error_{model}'].to_numpy() for model in models]
    plt.figure(figsize=(8.5, 5.5), dpi=140)
    plt.boxplot(data, labels=models, showfliers=False)
    plt.ylabel('Absolute Error (K)')
    plt.title(f'{split_name}: absolute error distribution by model')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances: pd.Series, save_path: Path, title: str) -> None:
    imp = importances.sort_values(ascending=True)
    plt.figure(figsize=(10, 7), dpi=140)
    plt.barh(imp.index, imp.values)
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# ----------------------------
# Model helpers
# ----------------------------
def fit_baseline_rf(X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=1000,
        max_features=10,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def tune_rf(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
) -> Tuple[RandomForestRegressor, Dict[str, float], pd.DataFrame]:
    _t = stage_start('Tune RF on validation split')
    grid = list(ParameterGrid({
        'n_estimators': [800, 1200, 1600],
        'max_features': [6, 8, 10, 12, 16],
        'min_samples_leaf': [1, 2, 4],
    }))
    rows = []
    best = None
    best_rmse = np.inf
    for i, params in enumerate(grid, start=1):
        model = RandomForestRegressor(**params, n_jobs=-1, random_state=seed)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        row = dict(params)
        row['val_RMSE'] = rmse(y_val, pred)
        row['val_MAE'] = float(mean_absolute_error(y_val, pred))
        row['val_R2'] = float(r2_score(y_val, pred))
        rows.append(row)
        if row['val_RMSE'] < best_rmse:
            best_rmse = row['val_RMSE']
            best = (model, row)
        progress(i, len(grid), prefix='rf grid search', every=5)
    stage_end('Tune RF on validation split', _t)
    return best[0], best[1], pd.DataFrame(rows).sort_values('val_RMSE')


def tune_xgb(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
) -> Tuple[XGBRegressor, Dict[str, float], pd.DataFrame]:
    _t = stage_start('Tune XGBoost on validation split')
    grid = list(ParameterGrid({
        'n_estimators': [400, 800, 1200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.03, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3],
        'reg_lambda': [1.0, 3.0],
    }))
    rows = []
    best = None
    best_rmse = np.inf
    for i, params in enumerate(grid, start=1):
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=seed,
            n_jobs=1,
            tree_method='hist',
            **params,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict(X_val)
        row = dict(params)
        row['val_RMSE'] = rmse(y_val, pred)
        row['val_MAE'] = float(mean_absolute_error(y_val, pred))
        row['val_R2'] = float(r2_score(y_val, pred))
        rows.append(row)
        if row['val_RMSE'] < best_rmse:
            best_rmse = row['val_RMSE']
            best = (model, row)
        progress(i, len(grid), prefix='xgb grid search', every=10)
    stage_end('Tune XGBoost on validation split', _t)
    return best[0], best[1], pd.DataFrame(rows).sort_values('val_RMSE')


def refit_best_model(best_name: str, best_params: Dict[str, float], X_train_full: pd.DataFrame, y_train_full: pd.Series, seed: int):
    if best_name == 'Baseline_RF':
        model = RandomForestRegressor(
            n_estimators=1000, max_features=10, min_samples_leaf=1,
            n_jobs=-1, random_state=seed,
        )
    elif best_name == 'Tuned_RF':
        params = {k: best_params[k] for k in ['n_estimators', 'max_features', 'min_samples_leaf']}
        model = RandomForestRegressor(**params, n_jobs=-1, random_state=seed)
    elif best_name == 'Tuned_XGB':
        params = {
            k: best_params[k]
            for k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'min_child_weight', 'reg_lambda']
        }
        model = XGBRegressor(objective='reg:squarederror', random_state=seed, n_jobs=1, tree_method='hist', **params)
    else:
        raise ValueError(f'Unknown model: {best_name}')
    model.fit(X_train_full, y_train_full)
    return model


def build_prediction_table(
    materials: pd.Series,
    y_true: pd.Series,
    pred_map: Dict[str, np.ndarray],
) -> pd.DataFrame:
    df = pd.DataFrame({
        'material': materials.reset_index(drop=True).to_numpy(),
        'true_Tc': y_true.reset_index(drop=True).to_numpy(),
    })
    for model_name, preds in pred_map.items():
        df[f'pred_{model_name}'] = preds
        df[f'abs_error_{model_name}'] = np.abs(df['true_Tc'] - df[f'pred_{model_name}'])
    return df


# ----------------------------
# Main pipeline
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description='Improve top24 model performance and produce defense plots.')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
    parser.add_argument('--final-tolerance', type=float, default=10.0)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    _t_all = stage_start('Improve top24 models and generate plots')

    train_df, unique_df, _ = load_aligned_datasets()
    X = train_df[TOP24_FEATURES].copy()
    y = train_df[TARGET_COLUMN].astype(float).copy()
    materials = unique_df['material'].astype(str).copy()

    log(f'Dataset loaded: X={X.shape}, y={y.shape}')

    X_train_full, X_test, y_train_full, y_test, mat_train_full, mat_test = train_test_split(
        X, y, materials, test_size=args.test_size, random_state=args.random_state
    )
    X_tr, X_val, y_tr, y_val, mat_tr, mat_val = train_test_split(
        X_train_full, y_train_full, mat_train_full, test_size=0.25, random_state=args.random_state
    )
    log(f'Outer split: train={len(X_train_full)}, test={len(X_test)} | inner validation train={len(X_tr)}, val={len(X_val)}')

    tolerances = [1, 5, 10]
    compare_rows = []

    # 1) Baseline RF
    _t = stage_start('Train baseline RF')
    baseline_rf = fit_baseline_rf(X_tr, y_tr, args.random_state)
    pred_val_baseline = baseline_rf.predict(X_val)
    baseline_metrics = metrics_dict(y_val.to_numpy(), pred_val_baseline, tolerances)
    compare_rows.append({'model': 'Baseline_RF', **baseline_metrics})
    stage_end('Train baseline RF', _t)

    # 2) Tuned RF
    tuned_rf_model, tuned_rf_best, tuned_rf_grid = tune_rf(X_tr, y_tr, X_val, y_val, args.random_state)
    pred_val_tuned_rf = tuned_rf_model.predict(X_val)
    compare_rows.append({'model': 'Tuned_RF', **metrics_dict(y_val.to_numpy(), pred_val_tuned_rf, tolerances)})
    tuned_rf_grid.to_csv(OUTDIR / 'rf_tuning_validation_results.csv', index=False)

    # 3) Tuned XGB
    tuned_xgb_model, tuned_xgb_best, tuned_xgb_grid = tune_xgb(X_tr, y_tr, X_val, y_val, args.random_state)
    pred_val_tuned_xgb = tuned_xgb_model.predict(X_val)
    compare_rows.append({'model': 'Tuned_XGB', **metrics_dict(y_val.to_numpy(), pred_val_tuned_xgb, tolerances)})
    tuned_xgb_grid.to_csv(OUTDIR / 'xgb_tuning_validation_results.csv', index=False)

    # Save validation model comparison table (existing output kept)
    validation_metrics_df = pd.DataFrame(compare_rows)
    validation_metrics_df.to_csv(OUTDIR / 'validation_model_comparison.csv', index=False)

    # NEW: more explicit RF/XGB same-validation comparison outputs
    validation_metrics_df.to_csv(OUTDIR / 'validation_rf_xgb_comparison_explicit.csv', index=False)

    validation_pred_df = build_prediction_table(
        materials=mat_val,
        y_true=y_val,
        pred_map={
            'Baseline_RF': pred_val_baseline,
            'Tuned_RF': pred_val_tuned_rf,
            'Tuned_XGB': pred_val_tuned_xgb,
        },
    )
    validation_pred_df.to_csv(OUTDIR / 'validation_predictions_all_models.csv', index=False)

    # Choose best on validation by RMSE
    best_row = validation_metrics_df.sort_values('RMSE').iloc[0].to_dict()
    best_name = best_row['model']
    log(f'Best validation model: {best_name} with RMSE={best_row["RMSE"]:.6f}')

    param_summary = {
        'Baseline_RF': {'n_estimators': 1000, 'max_features': 10, 'min_samples_leaf': 1},
        'Tuned_RF': tuned_rf_best,
        'Tuned_XGB': tuned_xgb_best,
    }
    with open(OUTDIR / 'best_model_selection.json', 'w', encoding='utf-8') as f:
        json.dump({'best_model': best_name, 'params': param_summary, 'test_size': args.test_size}, f, indent=2)

    # Refit ALL three models on full training set to provide explicit test-set comparison.
    _t = stage_start('Refit all three models on full training set and evaluate on held-out test set')
    baseline_rf_full = refit_best_model('Baseline_RF', param_summary['Baseline_RF'], X_train_full, y_train_full, args.random_state)
    tuned_rf_full = refit_best_model('Tuned_RF', param_summary['Tuned_RF'], X_train_full, y_train_full, args.random_state)
    tuned_xgb_full = refit_best_model('Tuned_XGB', param_summary['Tuned_XGB'], X_train_full, y_train_full, args.random_state)

    pred_test_baseline = baseline_rf_full.predict(X_test)
    pred_test_tuned_rf = tuned_rf_full.predict(X_test)
    pred_test_tuned_xgb = tuned_xgb_full.predict(X_test)
    stage_end('Refit all three models on full training set and evaluate on held-out test set', _t)

    # NEW: explicit test-set comparison among all three models
    test_compare_rows = [
        {'model': 'Baseline_RF', **metrics_dict(y_test.to_numpy(), pred_test_baseline, tolerances)},
        {'model': 'Tuned_RF', **metrics_dict(y_test.to_numpy(), pred_test_tuned_rf, tolerances)},
        {'model': 'Tuned_XGB', **metrics_dict(y_test.to_numpy(), pred_test_tuned_xgb, tolerances)},
    ]
    test_metrics_all_df = pd.DataFrame(test_compare_rows)
    test_metrics_all_df.to_csv(OUTDIR / 'test_model_comparison.csv', index=False)

    test_pred_df = build_prediction_table(
        materials=mat_test,
        y_true=y_test,
        pred_map={
            'Baseline_RF': pred_test_baseline,
            'Tuned_RF': pred_test_tuned_rf,
            'Tuned_XGB': pred_test_tuned_xgb,
        },
    )
    test_pred_df.to_csv(OUTDIR / 'test_predictions_all_models.csv', index=False)

    # Final chosen best model outputs (existing behavior kept)
    best_model = {
        'Baseline_RF': baseline_rf_full,
        'Tuned_RF': tuned_rf_full,
        'Tuned_XGB': tuned_xgb_full,
    }[best_name]
    y_test_pred = {
        'Baseline_RF': pred_test_baseline,
        'Tuned_RF': pred_test_tuned_rf,
        'Tuned_XGB': pred_test_tuned_xgb,
    }[best_name]

    test_metrics = metrics_dict(y_test.to_numpy(), y_test_pred, tolerances)
    save_predictions_csv(
        OUTDIR / 'best_model_test_predictions.csv',
        mat_test.reset_index(drop=True),
        y_test.to_numpy(),
        y_test_pred,
        args.final_tolerance,
        best_name,
        test_metrics,
    )

    pd.DataFrame([{'model': best_name, **test_metrics}]).to_csv(OUTDIR / 'best_model_test_metrics.csv', index=False)

    # Defense figures: keep old outputs
    plot_metric_comparison(validation_metrics_df, OUTDIR / 'validation_model_comparison.png')
    plot_tolerance_curve(y_test.to_numpy(), y_test_pred, OUTDIR / 'best_model_accuracy_vs_tolerance.png')
    plot_predicted_vs_true(y_test.to_numpy(), y_test_pred, OUTDIR / 'best_model_predicted_vs_true.png', f'{best_name}: predicted vs true on test set')
    plot_residuals(y_test.to_numpy(), y_test_pred, OUTDIR / 'best_model_residuals.png', f'{best_name}: residual diagnostics on test set')

    # NEW figures for explicit RF / XGB comparison and report usage
    plot_metric_comparison_full(validation_metrics_df, OUTDIR / 'validation_model_comparison_full_metrics.png', 'Validation')
    plot_metric_comparison_full(test_metrics_all_df, OUTDIR / 'test_model_comparison_full_metrics.png', 'Test')
    plot_predicted_vs_true_by_model(validation_pred_df, OUTDIR / 'validation_predicted_vs_true_three_models.png', 'Validation')
    plot_predicted_vs_true_by_model(test_pred_df, OUTDIR / 'test_predicted_vs_true_three_models.png', 'Test')
    plot_residual_histograms_by_model(validation_pred_df, OUTDIR / 'validation_residual_histograms_three_models.png', 'Validation')
    plot_residual_histograms_by_model(test_pred_df, OUTDIR / 'test_residual_histograms_three_models.png', 'Test')
    plot_absolute_error_boxplot(validation_pred_df, OUTDIR / 'validation_absolute_error_boxplot.png', 'Validation')
    plot_absolute_error_boxplot(test_pred_df, OUTDIR / 'test_absolute_error_boxplot.png', 'Test')

    if hasattr(best_model, 'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=TOP24_FEATURES)
        importances.sort_values(ascending=False).to_csv(OUTDIR / 'best_model_feature_importance.csv', header=['importance'])
        plot_feature_importance(importances, OUTDIR / 'best_model_feature_importance.png', f'{best_name}: top24 feature importance')

    summary_lines = [
        'How the metrics were improved:',
        '1) Use only the previously selected top24 features to reduce noise and redundant information.',
        '2) Keep a clean 80/20 hold-out split so evaluation is stricter and more interpretable.',
        '3) Compare a baseline RF against tuned RF and tuned XGBoost on the same validation split.',
        '4) Select the best model by validation RMSE, then retrain all three models on the full training split for explicit test-set comparison.',
        '5) Report MAE / RMSE / R2 together with accuracy under multiple tolerances (±1K, ±5K, ±10K).',
        '6) Export extra report/defense figures: full metric bars, predicted-vs-true panels, residual histograms, and absolute-error boxplots.',
        '',
        f'Best model selected: {best_name}',
        *[f'{k}: {v:.6f}' for k, v in test_metrics.items()],
        f'Final accuracy criterion written in predictions CSV: ±{args.final_tolerance:.2f}K',
    ]
    (OUTDIR / 'summary_explanation.txt').write_text('\n'.join(summary_lines), encoding='utf-8')

    stage_end('Improve top24 models and generate plots', _t_all)
    log(f'All outputs written to {OUTDIR}')


if __name__ == '__main__':
    main()
