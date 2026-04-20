from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

from feature_config import TOP24_FEATURES, ELEMENT_COLUMNS
from formula_parser import formula_to_vector

DATA_DIR = Path(__file__).resolve().parent / 'dataset'
TRAIN_CSV = DATA_DIR / 'train.csv'
UNIQUE_CSV = DATA_DIR / 'unique_m.csv'
TARGET = 'critical_temp'

BEST_XGB_PARAMS = {
    'n_estimators': 800,
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': 1,
    'tree_method': 'hist',
}


def _check_files() -> None:
    if not TRAIN_CSV.exists() or not UNIQUE_CSV.exists():
        raise FileNotFoundError(
            'Please put train.csv and unique_m.csv into the dataset/ folder next to the app.'
        )


@st.cache_data(show_spinner=False)
def load_train_df() -> pd.DataFrame:
    _check_files()
    return pd.read_csv(TRAIN_CSV)


@st.cache_data(show_spinner=False)
def load_unique_df() -> pd.DataFrame:
    _check_files()
    return pd.read_csv(UNIQUE_CSV)


@st.cache_resource(show_spinner=True)
def get_feature_model():
    df = load_train_df()
    X = df[TOP24_FEATURES].copy()
    y = df[TARGET].astype(float)
    model = XGBRegressor(**BEST_XGB_PARAMS)
    model.fit(X, y)
    return model


@st.cache_resource(show_spinner=True)
def get_formula_model():
    df = load_unique_df().copy()
    X = df[ELEMENT_COLUMNS].copy()
    y = df[TARGET].astype(float)
    model = XGBRegressor(**BEST_XGB_PARAMS)
    model.fit(X, y)

    nn = NearestNeighbors(n_neighbors=min(5, len(df)), metric='euclidean')
    nn.fit(X.values)
    return model, nn


@st.cache_data(show_spinner=False)
def get_metrics_snapshot() -> dict:
    return {
        'best_model': 'Tuned_XGB',
        'test_MAE': 5.1531,
        'test_RMSE': 8.8826,
        'test_R2': 0.9315,
        'ACC@±1K': 0.2680,
        'ACC@±5K': 0.6974,
        'ACC@±10K': 0.8467,
        'n_features': 24,
    }


def predict_from_formula(formula: str) -> tuple[float, pd.DataFrame]:
    model, nn = get_formula_model()
    df = load_unique_df()
    vec = formula_to_vector(formula)
    x = pd.DataFrame([vec], columns=ELEMENT_COLUMNS)
    pred = float(model.predict(x)[0])

    dist, idx = nn.kneighbors(x.values)
    neighbors = df.iloc[idx[0]][['material', TARGET]].copy()
    neighbors.insert(0, 'distance', dist[0])
    neighbors.rename(columns={TARGET: 'known_Tc'}, inplace=True)
    return pred, neighbors


def predict_from_top24(feature_values: dict) -> float:
    model = get_feature_model()
    x = pd.DataFrame([[feature_values[f] for f in TOP24_FEATURES]], columns=TOP24_FEATURES)
    return float(model.predict(x)[0])


def batch_predict_features(upload_df: pd.DataFrame) -> pd.DataFrame:
    model = get_feature_model()
    missing = [c for c in TOP24_FEATURES if c not in upload_df.columns]
    if missing:
        raise ValueError(f'Missing top24 columns: {missing}')
    out = upload_df.copy()
    out['predicted_Tc'] = model.predict(upload_df[TOP24_FEATURES])
    return out


def batch_predict_formula(upload_df: pd.DataFrame, formula_col: str = 'material') -> pd.DataFrame:
    model, _ = get_formula_model()
    out = upload_df.copy()
    vectors = []
    for f in out[formula_col].astype(str):
        vectors.append(formula_to_vector(f))
    X = pd.DataFrame(vectors, columns=ELEMENT_COLUMNS)
    out['predicted_Tc'] = model.predict(X)
    return out


def feature_importance_df() -> pd.DataFrame:
    model = get_feature_model()
    imp = pd.DataFrame({'feature': TOP24_FEATURES, 'importance': model.feature_importances_})
    return imp.sort_values('importance', ascending=False).reset_index(drop=True)


def sample_feature_defaults() -> dict:
    df = load_train_df()
    row = df.iloc[0]
    return {f: float(row[f]) for f in TOP24_FEATURES}


# def get_examples() -> list[str]:
#     df = load_unique_df()
#     examples = df['material'].astype(str).head(10).tolist()
#     return examples

def get_examples() -> list[str]:
    return [
        "Ir3Te8",
        "Nb3Se4",
        "La1Rh5",
        "Pb1Mo6Se8",
        "Al2La1",
        "Pb1Ta1Se2",
        "Ba1La1CuO4",
        "Pt3Y7",
        "Ca3Rh4Sn13",
        "Ag1Sn1Se2"
    ]