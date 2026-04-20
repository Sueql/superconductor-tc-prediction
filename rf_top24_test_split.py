
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import DATASET_DIR, OUTPUT_DIR, RANDOM_STATE, TARGET_COLUMN
from data_loader import load_aligned_datasets
from progress_utils import log, stage_start, stage_end

# Top-24 features selected in the previous RF-RFE + top-n selection stage.
TOP24_FEATURES: List[str] = [
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Split the dataset into 80%% train and 20%% test, retrain a RandomForest '
            'using the previously selected top-24 features, then write per-material '
            'test predictions to a CSV file.'
        )
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Fraction used as test set. Default: 0.2'
    )
    parser.add_argument(
        '--random-state', type=int, default=RANDOM_STATE,
        help=f'Random seed for the split and model. Default: {RANDOM_STATE}'
    )
    parser.add_argument(
        '--n-estimators', type=int, default=1000,
        help='Number of trees in RandomForestRegressor. Default: 1000'
    )
    parser.add_argument(
        '--max-features', type=int, default=10,
        help='max_features for RandomForestRegressor. Default: 10'
    )
    parser.add_argument(
        '--min-samples-leaf', type=int, default=1,
        help='min_samples_leaf for RandomForestRegressor. Default: 1'
    )
    parser.add_argument(
        '--tolerance', type=float, default=10.0,
        help=(
            'Absolute error tolerance (in K) used to decide Correct/Error in the output file. '
            'Default: 10.0 K'
        )
    )
    parser.add_argument(
        '--output', type=str,
        default=str(OUTPUT_DIR / 'rf_top24_holdout_test' / 'rf_top24_test_predictions.csv'),
        help='Path of the output CSV file.'
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    overall_t = stage_start('RF top24 80/20 holdout evaluation')

    log(f'Reading datasets from: {DATASET_DIR}')
    train_df, unique_df, train_with_indicators = load_aligned_datasets()
    log(f'Loaded train.csv shape={train_df.shape}, unique_m.csv shape={unique_df.shape}')

    missing = [c for c in TOP24_FEATURES if c not in train_df.columns]
    if missing:
        raise ValueError(f'The following top-24 features are missing in train.csv: {missing}')

    X = train_df[TOP24_FEATURES].copy()
    y = train_df[TARGET_COLUMN].astype(float).copy()
    materials = unique_df['material'].astype(str).copy()

    log('Creating aligned 80/20 train-test split')
    indices = np.arange(len(train_df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    materials_test = materials.iloc[test_idx].copy().reset_index(drop=True)

    log(f'Train size={len(train_idx)}, Test size={len(test_idx)}')
    log(f'Using top-24 features: {TOP24_FEATURES}')

    model_t = stage_start('Train RandomForestRegressor on top-24 features')
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    stage_end('Train RandomForestRegressor on top-24 features', model_t)

    eval_t = stage_start('Evaluate holdout test set')
    y_pred = model.predict(X_test)
    abs_error = np.abs(y_pred - y_test.to_numpy())
    result_flag = np.where(abs_error <= args.tolerance, 'Correct', 'Error')

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    accuracy = float(np.mean(result_flag == 'Correct'))

    results_df = pd.DataFrame({
        'material': materials_test,
        'true_Tc': y_test.to_numpy(),
        'predicted_Tc': y_pred,
        'prediction_result': result_flag,
    })

    summary_row = pd.DataFrame([{
        'material': 'OVERALL_TEST_ACCURACY',
        'true_Tc': np.nan,
        'predicted_Tc': np.nan,
        'prediction_result': (
            f'accuracy={accuracy:.6f}; tolerance={args.tolerance:.2f}K; '
            f'MAE={mae:.6f}; RMSE={rmse:.6f}; R2={r2:.6f}; '
            f'test_size={len(test_idx)}; train_size={len(train_idx)}'
        ),
    }])

    final_df = pd.concat([results_df, summary_row], ignore_index=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    log(f'Saved detailed test predictions to: {output_path.resolve()}')
    log(
        'Holdout summary | '
        f'accuracy(within ±{args.tolerance:.2f}K)={accuracy:.6f}, '
        f'MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}'
    )
    stage_end('Evaluate holdout test set', eval_t)
    stage_end('RF top24 80/20 holdout evaluation', overall_t)


if __name__ == '__main__':
    main()
