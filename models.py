from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    RANDOM_STATE,
    RF_FEATURE_METADATA_PATH,
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

    linear_coef = pd.Series(
        np.abs(linear.named_steps['model'].coef_),
        index=FEATURE_COLUMNS,
        name='abs_coefficient'
    ).sort_values(ascending=False)
    linear_coef.to_csv(outdir / 'linear_model_coef_size.csv', header=True)

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
        max_features=min(max_features, X.shape[1]),
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


def repeated_holdout_rf(train_df: pd.DataFrame, params: Dict[str, int], n_repeats: int = 25, seed: int = CV_RANDOM_STATE,
                        selected_features: List[str] | None = None) -> pd.DataFrame:
    X, y = get_feature_target(train_df)
    if selected_features is None:
        selected_features = list(X.columns)
    X = X[selected_features]
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(n_repeats):
        assign = sample_random_assignment(len(train_df), rng)
        test_mask = assign == 1
        X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
        y_train, y_test = y.loc[~test_mask], y.loc[test_mask]

        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_features=min(params['max_features'], len(selected_features)),
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
            'n_features': len(selected_features),
        })
    return pd.DataFrame(rows)


def _rf_for_selection(params: Dict[str, int], n_features: int, random_state: int, ranking_n_estimators: int | None = None) -> RandomForestRegressor:
    n_estimators = ranking_n_estimators if ranking_n_estimators is not None else params['n_estimators']
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=min(params['max_features'], n_features),
        min_samples_leaf=params['min_samples_leaf'],
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=random_state,
    )


def rf_rfe_ranking_from_xy(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, int],
    permutation_repeats: int = 3,
    ranking_n_estimators: int | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, list[str]]:
    current_features = list(X.columns)
    removed_rows: list[dict] = []
    step = 0

    while len(current_features) > 1:
        step += 1
        model = _rf_for_selection(params, len(current_features), random_state + step, ranking_n_estimators)
        model.fit(X[current_features], y)
        perm = permutation_importance(
            model,
            X[current_features],
            y,
            scoring='neg_root_mean_squared_error',
            n_repeats=permutation_repeats,
            random_state=random_state + step,
            n_jobs=-1,
        )
        imp = pd.Series(perm.importances_mean, index=current_features)
        least = imp.idxmin()
        removed_rows.append({
            'step': step,
            'n_features_before': len(current_features),
            'removed_feature': least,
            'importance_at_removal': float(imp.loc[least]),
        })
        current_features.remove(least)

    most_important = current_features[0]
    ranking_desc = [most_important] + [row['removed_feature'] for row in reversed(removed_rows)]
    ranking_df = pd.DataFrame(removed_rows)
    ranking_df['rank_from_most_important'] = ranking_df['removed_feature'].map(
        {feat: idx + 1 for idx, feat in enumerate(ranking_desc)}
    )
    return ranking_df, ranking_desc


def recursive_feature_elimination_rf(
    train_df: pd.DataFrame,
    outdir: Path,
    params: Dict[str, int] | None = None,
    permutation_repeats: int = 3,
    ranking_n_estimators: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    _ensure_dir(outdir)
    if params is None:
        params = {'max_features': 10, 'n_estimators': 500, 'min_samples_leaf': 1}

    X, y = get_feature_target(train_df)
    ranking_df, ranking_desc = rf_rfe_ranking_from_xy(
        X,
        y,
        params=params,
        permutation_repeats=permutation_repeats,
        ranking_n_estimators=ranking_n_estimators,
        random_state=RANDOM_STATE,
    )
    ranking_df.to_csv(outdir / 'rfe_variable_importance.csv', index=False)
    pd.DataFrame({'feature': ranking_desc, 'rank': np.arange(1, len(ranking_desc) + 1)}).to_csv(
        outdir / 'rfe_ranking_most_to_least.csv', index=False
    )

    top = ranking_desc[:20]
    plt.figure(figsize=(8, 6))
    plt.scatter(np.arange(len(top)), np.arange(len(top)), s=1, alpha=0)
    plt.yticks(np.arange(len(top)), top, fontsize=8)
    plt.xticks([])
    plt.title('RFE top 20 features (most important to less important)')
    plt.tight_layout()
    plt.savefig(outdir / 'rfe_variable_importance_top_20.png', dpi=220)
    plt.close()
    return ranking_df, ranking_desc


def _default_candidate_ns(total_features: int) -> list[int]:
    candidates = sorted(set([5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, total_features]))
    return [n for n in candidates if 1 <= n <= total_features]


def select_top_n_via_rfe_cv(
    train_df: pd.DataFrame,
    outdir: Path,
    params: Dict[str, int],
    candidate_ns: Iterable[int] | None = None,
    n_repeats: int = 5,
    permutation_repeats: int = 2,
    ranking_n_estimators: int | None = None,
    seed: int = CV_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    _ensure_dir(outdir)
    X, y = get_feature_target(train_df)
    rng = np.random.default_rng(seed)
    if candidate_ns is None:
        candidate_ns = _default_candidate_ns(X.shape[1])
    candidate_ns = sorted(set(int(n) for n in candidate_ns if 1 <= int(n) <= X.shape[1]))

    rows: list[dict] = []
    for repeat in range(n_repeats):
        assign = sample_random_assignment(len(train_df), rng)
        test_mask = assign == 1
        X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
        y_train, y_test = y.loc[~test_mask], y.loc[test_mask]

        _, ranking_desc = rf_rfe_ranking_from_xy(
            X_train,
            y_train,
            params=params,
            permutation_repeats=permutation_repeats,
            ranking_n_estimators=ranking_n_estimators,
            random_state=seed + repeat * 1000,
        )

        for n in candidate_ns:
            selected = ranking_desc[:n]
            model = RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_features=min(params['max_features'], n),
                min_samples_leaf=params['min_samples_leaf'],
                bootstrap=True,
                oob_score=False,
                n_jobs=-1,
                random_state=seed + repeat,
            )
            model.fit(X_train[selected], y_train)
            pred = model.predict(X_test[selected])
            rows.append({
                'repeat': repeat + 1,
                'n_features': n,
                'rmse': rmse(y_test, pred),
                'r2': float(r2_score(y_test, pred)),
            })

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(outdir / 'rfe_topn_cv_detail.csv', index=False)

    summary_df = detail_df.groupby('n_features')[['rmse', 'r2']].agg(['mean', 'std']).reset_index()
    summary_df.columns = ['n_features', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std']
    summary_df = summary_df.sort_values(['rmse_mean', 'n_features'], ascending=[True, True])
    summary_df.to_csv(outdir / 'rfe_topn_cv_summary.csv', index=False)

    best_n = int(summary_df.iloc[0]['n_features'])

    plt.figure(figsize=(8, 5))
    plt.errorbar(summary_df['n_features'], summary_df['rmse_mean'], yerr=summary_df['rmse_std'], marker='o')
    plt.axvline(best_n, color='red', linestyle='--', label=f'best n={best_n}')
    plt.xlabel('Top-n features from RF-RFE ranking')
    plt.ylabel('RMSE (repeated holdout mean ± std)')
    plt.title('Selecting the best number of RF-RFE features')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'rfe_topn_selection_curve.png', dpi=220)
    plt.close()
    return detail_df, summary_df, best_n


def fit_final_random_forest(
    train_df: pd.DataFrame,
    outdir: Path,
    params: Dict[str, int] | None = None,
    selected_features: List[str] | None = None,
) -> Tuple[RandomForestRegressor, pd.DataFrame]:
    _ensure_dir(outdir)
    X, y = get_feature_target(train_df)
    if params is None:
        params = {'max_features': 10, 'n_estimators': 1000, 'min_samples_leaf': 1}
    if selected_features is None:
        selected_features = list(X.columns)

    X_sel = X[selected_features]
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_features=min(params['max_features'], len(selected_features)),
        min_samples_leaf=params['min_samples_leaf'],
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_sel, y)
    joblib.dump(model, RF_FEATURE_MODEL_PATH)

    metadata = {
        'selected_features': selected_features,
        'n_selected_features': len(selected_features),
        'rf_params': params,
        'selection_method': 'RF-RFE ranking + repeated holdout top-n selection' if len(selected_features) < len(FEATURE_COLUMNS) else 'all features',
    }
    RF_FEATURE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    oob_pred = model.oob_prediction_
    train_metrics = pd.DataFrame([{
        'n_selected_features': len(selected_features),
        'rmse_oob': rmse(y, oob_pred),
        'r2_oob': float(r2_score(y, oob_pred)),
        'rmse_in_sample': rmse(y, model.predict(X_sel)),
        'r2_in_sample': float(r2_score(y, model.predict(X_sel))),
    }])
    train_metrics.to_csv(outdir / 'rf_final_metrics.csv', index=False)

    imp = pd.Series(model.feature_importances_, index=selected_features).sort_values()
    imp.to_csv(outdir / 'rf_variable_importance.csv', header=['importance'])

    plt.figure(figsize=(8, max(6, len(imp) * 0.18)))
    plt.scatter(imp.values, np.arange(len(imp)), s=8)
    plt.yticks(np.arange(len(imp)), imp.index, fontsize=7)
    plt.xlabel('Impurity-based importance')
    plt.title('Random forest variable importance (selected features only)')
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
