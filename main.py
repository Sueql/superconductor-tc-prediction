from __future__ import annotations

import argparse
import json

import pandas as pd

from analysis import run_all_analyses
from config import FEATURE_COLUMNS, OUTPUT_DIR, RF_FEATURE_METADATA_PATH
from data_loader import load_aligned_datasets
from models import (
    fit_final_random_forest,
    run_linear_baseline_cv,
    run_optional_gbm_grid,
    train_formula_random_forest,
    train_full_linear_models,
    tune_random_forest,
    repeated_holdout_rf,
    recursive_feature_elimination_rf,
    select_top_n_via_rfe_cv,
)
from predictor import SuperconductorPredictor
from training import run_full_pipeline


def _load_best_params() -> dict:
    tuning_path = OUTPUT_DIR / 'rf_tuning' / 'rf_tuning_results.csv'
    if tuning_path.exists():
        tuning = pd.read_csv(tuning_path)
        best = tuning.iloc[0]
        return {
            'max_features': int(best['max_features']),
            'n_estimators': int(best['n_estimators']),
            'min_samples_leaf': int(best['min_samples_leaf']),
        }
    return {'max_features': 10, 'n_estimators': 1000, 'min_samples_leaf': 1}


def cmd_check_data() -> None:
    train_df, unique_df, train_with_indicators = load_aligned_datasets()
    print('train.csv:', train_df.shape)
    print('unique_m.csv:', unique_df.shape)
    print('train_with_indicators:', train_with_indicators.shape)


def cmd_analyze() -> None:
    train_df, unique_df, train_with_indicators = load_aligned_datasets()
    run_all_analyses(train_df, unique_df, train_with_indicators, OUTPUT_DIR / 'analysis')
    print('Analysis completed. Files saved to outputs/analysis/')


def cmd_train_baselines() -> None:
    train_df, _, _ = load_aligned_datasets()
    train_full_linear_models(train_df, OUTPUT_DIR / 'linear_full_fit')
    cv_df = run_linear_baseline_cv(train_df, OUTPUT_DIR / 'linear_cv')
    print(cv_df.groupby('model')[['rmse', 'r2']].mean())


def cmd_tune_rf() -> None:
    train_df, _, _ = load_aligned_datasets()
    results = tune_random_forest(train_df, OUTPUT_DIR / 'rf_tuning')
    print(results.head())


def cmd_train_rf() -> None:
    train_df, unique_df, _ = load_aligned_datasets()
    params = _load_best_params()

    ranking_df, ranking_desc = recursive_feature_elimination_rf(
        train_df,
        OUTPUT_DIR / 'rfe',
        params,
        permutation_repeats=3,
        ranking_n_estimators=min(300, params['n_estimators']),
    )
    _, summary_df, best_n = select_top_n_via_rfe_cv(
        train_df,
        OUTPUT_DIR / 'rfe_topn_selection',
        params,
        n_repeats=5,
        permutation_repeats=2,
        ranking_n_estimators=min(200, params['n_estimators']),
    )
    selected_features = ranking_desc[:best_n]

    rf_cv = repeated_holdout_rf(train_df, params, selected_features=selected_features)
    rf_cv.to_csv(OUTPUT_DIR / 'rf_cv_results.csv', index=False)
    print(rf_cv[['rmse', 'r2']].describe())

    fit_final_random_forest(train_df, OUTPUT_DIR / 'rf_final', params, selected_features)
    train_formula_random_forest(unique_df, OUTPUT_DIR / 'formula_model', params)
    print('Random forest models trained and saved.')
    print(f'Selected top-n features: {best_n}')
    print(summary_df.head())


def cmd_rfe() -> None:
    train_df, _, _ = load_aligned_datasets()
    params = _load_best_params()
    ranking_df, ranking_desc = recursive_feature_elimination_rf(
        train_df,
        OUTPUT_DIR / 'rfe',
        params,
        permutation_repeats=3,
        ranking_n_estimators=min(300, params['n_estimators']),
    )
    _, summary_df, best_n = select_top_n_via_rfe_cv(
        train_df,
        OUTPUT_DIR / 'rfe_topn_selection',
        params,
        n_repeats=5,
        permutation_repeats=2,
        ranking_n_estimators=min(200, params['n_estimators']),
    )
    print(ranking_df.tail())
    print('\nBest n according to repeated holdout CV:', best_n)
    print(summary_df.head())


def cmd_gbm() -> None:
    train_df, _, _ = load_aligned_datasets()
    results = run_optional_gbm_grid(train_df, OUTPUT_DIR / 'gbm_optional')
    print(results.head())


def cmd_train_all(with_gbm: bool) -> None:
    summary = run_full_pipeline(run_optional_gbm=with_gbm, run_rfe=True, rfe_max_steps=25)
    print(json.dumps(summary, indent=2))


def cmd_predict_formula(formula: str, match_level: float) -> None:
    predictor = SuperconductorPredictor()
    res = predictor.predict_from_formula(formula, match_level=match_level)
    print(f'Formula: {res.formula}')
    print(f'Predicted Tc: {res.predicted_tc:.4f} K')
    print('\nExact/near-exact matches:')
    print(res.exact_matches if not res.exact_matches.empty else 'None')
    print('\nMost similar materials:')
    print(res.similar_materials)


def cmd_predict_feature_row(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    if len(df) != 1:
        raise ValueError('Input CSV must contain exactly one row.')
    predictor = SuperconductorPredictor()
    pred = predictor.predict_from_feature_row(df.iloc[0].to_dict())
    print(f'Predicted Tc: {pred:.4f} K')
    if RF_FEATURE_METADATA_PATH.exists():
        metadata = json.loads(RF_FEATURE_METADATA_PATH.read_text(encoding='utf-8'))
        print('Selected feature count:', metadata.get('n_selected_features'))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Superconductor Tc prediction project')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('check-data')
    sub.add_parser('analyze')
    sub.add_parser('train-baselines')
    sub.add_parser('tune-rf')
    sub.add_parser('train-rf')
    sub.add_parser('rfe')
    sub.add_parser('gbm')

    p_all = sub.add_parser('train-all')
    p_all.add_argument('--with-gbm', action='store_true')

    p_formula = sub.add_parser('predict-formula')
    p_formula.add_argument('--formula', required=True)
    p_formula.add_argument('--match-level', type=float, default=0.999999)

    p_feat = sub.add_parser('predict-feature-row')
    p_feat.add_argument('--csv', required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'check-data':
        cmd_check_data()
    elif args.command == 'analyze':
        cmd_analyze()
    elif args.command == 'train-baselines':
        cmd_train_baselines()
    elif args.command == 'tune-rf':
        cmd_tune_rf()
    elif args.command == 'train-rf':
        cmd_train_rf()
    elif args.command == 'rfe':
        cmd_rfe()
    elif args.command == 'gbm':
        cmd_gbm()
    elif args.command == 'train-all':
        cmd_train_all(args.with_gbm)
    elif args.command == 'predict-formula':
        cmd_predict_formula(args.formula, args.match_level)
    elif args.command == 'predict-feature-row':
        cmd_predict_feature_row(args.csv)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
