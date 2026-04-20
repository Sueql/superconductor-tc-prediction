from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, RANDOM_STATE
from data_loader import load_aligned_datasets

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Split train.csv into 80%% train / 20%% test, train a new XGBoost model '
            'using the pre-selected top 24 features, and evaluate on the test split.'
        )
    )
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio, default=0.2')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE, help='Random seed, default=42')
    parser.add_argument('--tolerance', type=float, default=10.0, help='Correct if |pred-true| <= tolerance, default=10.0 K')
    parser.add_argument('--n-estimators', type=int, default=1200)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.03)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    parser.add_argument('--outdir', type=str, default=str(OUTPUT_DIR / 'xgb_top24_holdout_test'))
    return parser.parse_args()


def _require_xgboost():
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            'This script needs xgboost. Install it first with: pip install xgboost\n'
            f'Original import error: {exc}'
        )
    return XGBRegressor


def build_output_dataframe(materials: pd.Series, y_true: pd.Series, y_pred: np.ndarray, tolerance: float) -> pd.DataFrame:
    abs_err = np.abs(y_pred - y_true.to_numpy(dtype=float))
    correct = abs_err <= tolerance
    out = pd.DataFrame(
        {
            'material': materials.to_numpy(),
            'true_Tc': y_true.to_numpy(dtype=float),
            'predicted_Tc': y_pred.astype(float),
            'prediction_result': np.where(correct, 'Correct', 'Error'),
        }
    )
    return out


def append_summary_row(df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, tolerance: float, train_size: int, test_size: int) -> pd.DataFrame:
    abs_err = np.abs(y_pred - y_true.to_numpy(dtype=float))
    accuracy = float(np.mean(abs_err <= tolerance))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    summary_text = (
        f'accuracy={accuracy:.6f}; tolerance={tolerance:.2f}K; '
        f'MAE={mae:.6f}; RMSE={rmse:.6f}; R2={r2:.6f}; '
        f'test_size={test_size}; train_size={train_size}'
    )
    summary_row = pd.DataFrame(
        [
            {
                'material': 'OVERALL_TEST_ACCURACY',
                'true_Tc': np.nan,
                'predicted_Tc': np.nan,
                'prediction_result': summary_text,
            }
        ]
    )
    return pd.concat([df, summary_row], ignore_index=True)


def main() -> None:
    args = parse_args()
    XGBRegressor = _require_xgboost()

    print('Step 1/5: Loading aligned datasets ...', flush=True)
    train_df, unique_df, _ = load_aligned_datasets()

    print('Step 2/5: Building feature matrix using the fixed top 24 features ...', flush=True)
    X = train_df[TOP24_FEATURES].copy()
    y = train_df['critical_temp'].astype(float).copy()
    materials = unique_df['material'].astype(str).copy()

    print('Step 3/5: Splitting into 80% train / 20% test ...', flush=True)
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        X,
        y,
        materials,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    print('Step 4/5: Training XGBoost on the training split ...', flush=True)
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective='reg:squarederror',
        random_state=args.random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    print('Step 5/5: Testing XGBoost on the held-out test split ...', flush=True)
    y_pred = model.predict(X_test)

    result_df = build_output_dataframe(m_test, y_test, y_pred, tolerance=args.tolerance)
    result_df = append_summary_row(
        result_df,
        y_true=y_test,
        y_pred=y_pred,
        tolerance=args.tolerance,
        train_size=len(X_train),
        test_size=len(X_test),
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / 'xgb_top24_test_predictions.csv'
    result_df.to_csv(csv_path, index=False)

    # Also save a small metrics file for convenience.
    abs_err = np.abs(y_pred - y_test.to_numpy(dtype=float))
    metrics_df = pd.DataFrame(
        [
            {
                'accuracy_within_tolerance': float(np.mean(abs_err <= args.tolerance)),
                'tolerance_K': float(args.tolerance),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'r2': float(r2_score(y_test, y_pred)),
                'train_size': int(len(X_train)),
                'test_size': int(len(X_test)),
                'n_features': int(len(TOP24_FEATURES)),
            }
        ]
    )
    metrics_df.to_csv(outdir / 'xgb_top24_test_metrics.csv', index=False)

    print(f'Done. Results saved to: {csv_path}', flush=True)



if __name__ == '__main__':
    main()
