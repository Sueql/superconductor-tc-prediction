from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CV_RANDOM_STATE,
    FEATURE_COLUMNS,
    FORMULA_COLUMN,
    LINEAR_MODEL_PATH,
    MODEL_DIR,
    RANDOM_STATE,
    RF_FEATURE_MODEL_PATH,
    RF_FORMULA_MODEL_PATH,
    RIDGE_MODEL_PATH,
    TARGET_COLUMN,
)
from data_loader import get_feature_target, get_formula_target, sample_random_assignment


@dataclass
class EvalResult:
    rmse: float
    r2: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> EvalResult:
    pred = model.predict(X_test)
    return EvalResult(rmse=rmse(y_test, pred), r2=float(r2_score(y_test, pred)))


def make_linear_pipeline() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression()),
    ])


def make_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=alpha)),
    ])


def train_full_linear_models(train_df: pd.DataFrame, outdir: Path) -> Dict[str, EvalResult]:
    _ensure_dir(outdir)
    X, y = get_feature_target(train_df)

    linear = make_linear_pipeline().fit(X, y)
    ridge = make_ridge_pipeline(1.0).fit(X, y)

    joblib.dump(linear, LINEAR_MODEL_PATH)
    joblib.dump(ridge, RIDGE_MODEL_PATH)

    linear_pred = linear.predict(X)
    ridge_pred = ridge.predict(X)

    metrics = {
        'LinearRegression_train': EvalResult(rmse(y, linear_pred), float(r2_score(y, linear_pred))),
        'Ridge_train': EvalResult(rmse(y, ridge_pred), float(r2_score(y, ridge_pred))),
    }

    # Coefficient report
    linear_coef = pd.Series(
        np.abs(linear.named_steps['model'].coef_),
        index=FEATURE_COLUMNS,
        name='abs_coefficient'
    ).sort_values(ascending=False)
    linear_coef.to_csv(outdir / 'linear_model_coef_size.csv', header=True)

    # Predicted vs observed + residuals for full-data fit
    for name, pred in [('linear', linear_pred), ('ridge', ridge_pred)]:
        plt.figure(figsize=(6, 5))
        plt.scatter(y, pred, s=8, alpha=0.45)
        lo = float(min(y.min(), pred.min()))
        hi = float(max(y.max(), pred.max()))
        plt.plot([lo, hi], [lo, hi], 'r--')
        plt.xlabel('Observed critical temperature (K)')
        plt.ylabel('Predicted critical temperature (K)')
        plt.title(f'{name.capitalize()} predicted vs observed')
        plt.tight_layout()
        plt.savefig(outdir / f'{name}_predicted_vs_observed.png', dpi=220)
        plt.close()

        resid = y - pred
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].scatter(y, resid, s=8, alpha=0.45)
        axes[0].axhline(0, linestyle='--', color='gray')
        axes[0].set_xlabel('Observed critical temperature (K)')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs observed')
        axes[1].hist(resid, bins=40, color='lightgray', edgecolor='black', density=True)
        axes[1].set_title('Residual histogram')
        axes[2].plot(np.sort(resid.to_numpy()), np.linspace(0, 1, len(resid)))
        axes[2].set_title('Residual empirical CDF')
        plt.tight_layout()
        plt.savefig(outdir / f'{name}_residual_diagnostics.png', dpi=220)
        plt.close(fig)

    return metrics


def repeated_holdout_cv(train_df: pd.DataFrame, n_repeats: int = 25, seed: int = CV_RANDOM_STATE) -> pd.DataFrame:
    X, y = get_feature_target(train_df)
    rng = np.random.default_rng(seed)

    rows = []
    for repeat in range(n_repeats):
        assign = sample_random_assignment(len(train_df), rng)
        test_mask = assign == 1
        X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
        y_train, y_test = y.loc[~test_mask], y.loc[test_mask]

        linear = make_linear_pipeline().fit(X_train, y_train)
        ridge = make_ridge_pipeline(1.0).fit(X_train, y_train)

        for name, model in [('LinearRegression', linear), ('Ridge', ridge)]:
            result = evaluate_model(model, X_test, y_test)
            rows.append({
                'repeat': repeat + 1,
                'model': name,
                'rmse': result.rmse,
                'r2': result.r2,
                'test_size': len(X_test),
            })
    return pd.DataFrame(rows)


def run_linear_baseline_cv(train_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    _ensure_dir(outdir)
    cv_df = repeated_holdout_cv(train_df)
    cv_df.to_csv(outdir / 'linear_ridge_cv_results.csv', index=False)

    summary = cv_df.groupby('model')[['rmse', 'r2']].describe()
    summary.to_csv(outdir / 'linear_ridge_cv_summary.csv')

    plt.figure(figsize=(8, 4))
    for i, metric in enumerate(['rmse', 'r2'], start=1):
        plt.subplot(1, 2, i)
        for model_name, grp in cv_df.groupby('model'):
            plt.plot(np.sort(grp[metric].to_numpy()), marker='o', ms=3, label=model_name)
        plt.title(metric.upper())
        plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'linear_ridge_cv_sorted_metrics.png', dpi=220)
    plt.close()
    return cv_df


def _fit_rf_oob(X: pd.DataFrame, y: pd.Series, *, max_features: int, n_estimators: int, min_samples_leaf: int, random_state: int) -> Tuple[RandomForestRegressor, float]:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X, y)
    if not hasattr(model, 'oob_prediction_'):
        raise RuntimeError('OOB predictions are unavailable. Check bootstrap/oob settings.')
    score = rmse(y, model.oob_prediction_)
    return model, score


def tune_random_forest(train_df: pd.DataFrame, outdir: Path, max_features_grid: Iterable[int] | None = None,
                       n_estimators_grid: Iterable[int] = (1000, 2500),
                       min_samples_leaf_grid: Iterable[int] = (1, 5, 25)) -> pd.DataFrame:
    _ensure_dir(outdir)
    X, y = get_feature_target(train_df)
    if max_features_grid is None:
        max_features_grid = range(1, min(40, X.shape[1]) + 1)

    rows = []
    for max_features in max_features_grid:
        for n_estimators in n_estimators_grid:
            for min_leaf in min_samples_leaf_grid:
                _, score = _fit_rf_oob(
                    X, y,
                    max_features=max_features,
                    n_estimators=n_estimators,
                    min_samples_leaf=min_leaf,
                    random_state=RANDOM_STATE,
                )
                rows.append({
                    'max_features': max_features,
                    'n_estimators': n_estimators,
                    'min_samples_leaf': min_leaf,
                    'oob_rmse': score,
                })
    results = pd.DataFrame(rows).sort_values('oob_rmse')
    results.to_csv(outdir / 'rf_tuning_results.csv', index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(np.sort(results['oob_rmse'].to_numpy()))
    plt.ylabel('OOB RMSE')
    plt.xlabel('Grid point rank')
    plt.title('Random forest tuning results')
    plt.tight_layout()
    plt.savefig(outdir / 'rf_tuning_sorted_rmse.png', dpi=220)
    plt.close()
    return results


def repeated_holdout_rf(train_df: pd.DataFrame, params: Dict[str, int], n_repeats: int = 25, seed: int = CV_RANDOM_STATE) -> pd.DataFrame:
    X, y = get_feature_target(train_df)
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(n_repeats):
        assign = sample_random_assignment(len(train_df), rng)
        test_mask = assign == 1
        X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
        y_train, y_test = y.loc[~test_mask], y.loc[test_mask]

        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_features=params['max_features'],
            min_samples_leaf=params['min_samples_leaf'],
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=seed + repeat,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            'repeat': repeat + 1,
            'rmse': rmse(y_test, pred),
            'r2': float(r2_score(y_test, pred)),
            'test_size': len(X_test),
        })
    return pd.DataFrame(rows)


def fit_final_random_forest(train_df: pd.DataFrame, outdir: Path, params: Dict[str, int] | None = None) -> Tuple[RandomForestRegressor, pd.DataFrame]:
    _ensure_dir(outdir)
    X, y = get_feature_target(train_df)
    if params is None:
        params = {'max_features': 10, 'n_estimators': 1000, 'min_samples_leaf': 1}

    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_features=params['max_features'],
        min_samples_leaf=params['min_samples_leaf'],
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    joblib.dump(model, RF_FEATURE_MODEL_PATH)

    oob_pred = model.oob_prediction_
    train_metrics = pd.DataFrame([{
        'rmse_oob': rmse(y, oob_pred),
        'r2_oob': float(r2_score(y, oob_pred)),
        'rmse_in_sample': rmse(y, model.predict(X)),
        'r2_in_sample': float(r2_score(y, model.predict(X))),
    }])
    train_metrics.to_csv(outdir / 'rf_final_metrics.csv', index=False)

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    imp.to_csv(outdir / 'rf_variable_importance.csv', header=['importance'])

    plt.figure(figsize=(8, 12))
    plt.scatter(imp.values, np.arange(len(imp)), s=8)
    plt.yticks(np.arange(len(imp)), imp.index, fontsize=7)
    plt.xlabel('Impurity-based importance')
    plt.title('Random forest variable importance')
    plt.tight_layout()
    plt.savefig(outdir / 'random_forest_variable_importance.png', dpi=220)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(y, oob_pred, s=8, alpha=0.45)
    lo = float(min(y.min(), oob_pred.min()))
    hi = float(max(y.max(), oob_pred.max()))
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.xlabel('Observed critical temperature (K)')
    plt.ylabel('Predicted critical temperature (K)')
    plt.title('Random forest OOB predicted vs observed')
    plt.tight_layout()
    plt.savefig(outdir / 'random_forest_oob_predicted_vs_observed.png', dpi=220)
    plt.close()

    residuals = y - oob_pred
    sd_limit = residuals.std(ddof=1)
    plt.figure(figsize=(6, 5))
    plt.scatter(y, residuals, s=8, alpha=0.45)
    plt.axhline(0, color='black')
    plt.axhline(sd_limit, color='red', linestyle='--')
    plt.axhline(-sd_limit, color='red', linestyle='--')
    plt.xlabel('Observed critical temperature (K)')
    plt.ylabel('Residuals (Observed - Predicted)')
    plt.title('Random forest residuals vs observed')
    plt.tight_layout()
    plt.savefig(outdir / 'random_forest_residual_vs_observed.png', dpi=220)
    plt.close()

    coverage = float((np.abs(residuals) <= sd_limit).mean())
    pd.DataFrame([{'residual_sd': float(sd_limit), 'within_1sd_fraction': coverage}]).to_csv(
        outdir / 'random_forest_residual_summary.csv', index=False
    )
    return model, train_metrics


def recursive_feature_elimination_rf(train_df: pd.DataFrame, outdir: Path, params: Dict[str, int] | None = None,
                                     n_repeats: int = 3, max_steps: int | None = None) -> pd.DataFrame:
    _ensure_dir(outdir)
    if params is None:
        params = {'max_features': 10, 'n_estimators': 500, 'min_samples_leaf': 1}

    X, y = get_feature_target(train_df)
    current_features = list(X.columns)
    removed_rows = []
    step = 0

    while len(current_features) > 1:
        step += 1
        if max_steps is not None and step > max_steps:
            break
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_features=min(params['max_features'], len(current_features)),
            min_samples_leaf=params['min_samples_leaf'],
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=RANDOM_STATE + step,
        )
        model.fit(X[current_features], y)
        perm = permutation_importance(
            model,
            X[current_features],
            y,
            scoring='neg_root_mean_squared_error',
            n_repeats=n_repeats,
            random_state=RANDOM_STATE + step,
            n_jobs=-1,
        )
        imp = pd.Series(perm.importances_mean, index=current_features)
        least = imp.idxmin()
        removed_rows.append({
            'step': step,
            'removed_feature': least,
            'importance': float(imp.loc[least]),
            'remaining_features': len(current_features) - 1,
        })
        current_features.remove(least)

    rfe_df = pd.DataFrame(removed_rows)
    rfe_df.to_csv(outdir / 'rfe_variable_importance.csv', index=False)
    if not rfe_df.empty:
        top = rfe_df.tail(min(20, len(rfe_df)))
        plt.figure(figsize=(8, 6))
        plt.scatter(top['importance'], np.arange(len(top)), s=18)
        plt.yticks(np.arange(len(top)), top['removed_feature'], fontsize=8)
        plt.xlabel('Permutation importance at removal')
        plt.title('RFE top removed features (late-stage = more important)')
        plt.tight_layout()
        plt.savefig(outdir / 'rfe_variable_importance_top_20.png', dpi=220)
        plt.close()
    return rfe_df


def run_optional_gbm_grid(train_df: pd.DataFrame, outdir: Path,
                          depths: Iterable[int] = (10, 12, 14, 16, 18),
                          learning_rates: Iterable[float] = (0.05, 0.10),
                          n_estimators_map: Dict[float, int] | None = None) -> pd.DataFrame:
    _ensure_dir(outdir)
    X, y = get_feature_target(train_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/3.0, random_state=10_000)
    if n_estimators_map is None:
        n_estimators_map = {0.05: 2000, 0.10: 1000}

    rows = []
    for lr in learning_rates:
        for depth in depths:
            model = GradientBoostingRegressor(
                random_state=RANDOM_STATE,
                learning_rate=lr,
                n_estimators=n_estimators_map[lr],
                max_depth=depth,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rows.append({
                'learning_rate': lr,
                'depth': depth,
                'n_estimators': n_estimators_map[lr],
                'rmse_test': rmse(y_test, pred),
                'r2_test': float(r2_score(y_test, pred)),
            })
    results = pd.DataFrame(rows).sort_values('rmse_test')
    results.to_csv(outdir / 'gbm_grid_results.csv', index=False)
    return results


def train_formula_random_forest(unique_df: pd.DataFrame, outdir: Path,
                                params: Dict[str, int] | None = None) -> RandomForestRegressor:
    _ensure_dir(outdir)
    X, y = get_formula_target(unique_df)
    if params is None:
        params = {'max_features': 10, 'n_estimators': 1000, 'min_samples_leaf': 1}
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_features=min(params['max_features'], X.shape[1]),
        min_samples_leaf=params['min_samples_leaf'],
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    joblib.dump(model, RF_FORMULA_MODEL_PATH)
    pd.DataFrame([{
        'rmse_oob': rmse(y, model.oob_prediction_),
        'r2_oob': float(r2_score(y, model.oob_prediction_)),
    }]).to_csv(outdir / 'rf_formula_metrics.csv', index=False)
    return model
