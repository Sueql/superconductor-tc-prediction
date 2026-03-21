from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    ELEMENTS,
    FEATURE_COLUMNS,
    FORMULA_COLUMN,
    TARGET_COLUMN,
    TRAIN_CSV,
    UNIQUE_M_CSV,
)


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')


def load_train_csv(path: Path = TRAIN_CSV) -> pd.DataFrame:
    _assert_exists(path)
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f'train.csv is missing columns: {missing[:10]}')
    return df.copy()


def load_unique_m_csv(path: Path = UNIQUE_M_CSV) -> pd.DataFrame:
    _assert_exists(path)
    df = pd.read_csv(path)
    required = ELEMENTS + [TARGET_COLUMN, FORMULA_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'unique_m.csv is missing columns: {missing[:10]}')
    return df.copy()


def add_formula_indicators(unique_df: pd.DataFrame) -> pd.DataFrame:
    out = unique_df.copy()
    out['iron'] = (out['Fe'] > 0).astype(int)
    out['cuprate'] = ((out['Cu'] > 0) & (out['O'] > 0)).astype(int)
    return out


def load_aligned_datasets(train_path: Path = TRAIN_CSV, unique_path: Path = UNIQUE_M_CSV) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = load_train_csv(train_path)
    unique_df = add_formula_indicators(load_unique_m_csv(unique_path))

    if len(train_df) != len(unique_df):
        raise ValueError(
            'train.csv and unique_m.csv row counts do not match. '
            'This project assumes they are aligned row-by-row as in the original R workflow.'
        )

    train_with_indicators = train_df.copy()
    train_with_indicators['iron'] = unique_df['iron'].to_numpy()
    train_with_indicators['cuprate'] = unique_df['cuprate'].to_numpy()
    train_with_indicators[FORMULA_COLUMN] = unique_df[FORMULA_COLUMN].to_numpy()
    return train_df, unique_df, train_with_indicators


def get_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return X, y


def get_formula_target(unique_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = unique_df[ELEMENTS].astype(float).copy()
    y = unique_df[TARGET_COLUMN].astype(float).copy()
    return X, y


def sample_random_assignment(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(low=1, high=4, size=n_rows)
