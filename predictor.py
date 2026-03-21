from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from config import (
    ELEMENTS,
    FEATURE_COLUMNS,
    FORMULA_COLUMN,
    RF_FEATURE_METADATA_PATH,
    RF_FEATURE_MODEL_PATH,
    RF_FORMULA_MODEL_PATH,
    TARGET_COLUMN,
)
from data_loader import load_aligned_datasets
from formula_parser import formula_to_vector, normalize_vector


@dataclass
class FormulaPredictionResult:
    formula: str
    predicted_tc: float
    exact_matches: pd.DataFrame
    similar_materials: pd.DataFrame


class SuperconductorPredictor:
    def __init__(self, feature_model_path: Path = RF_FEATURE_MODEL_PATH, formula_model_path: Path = RF_FORMULA_MODEL_PATH):
        self.feature_model = joblib.load(feature_model_path) if Path(feature_model_path).exists() else None
        self.formula_model = joblib.load(formula_model_path) if Path(formula_model_path).exists() else None
        self.train_df, self.unique_df, self.train_with_indicators = load_aligned_datasets()
        self.selected_features = list(FEATURE_COLUMNS)
        if RF_FEATURE_METADATA_PATH.exists():
            metadata = json.loads(RF_FEATURE_METADATA_PATH.read_text(encoding='utf-8'))
            self.selected_features = metadata.get('selected_features', list(FEATURE_COLUMNS))

    def predict_from_feature_row(self, feature_row: Dict[str, float]) -> float:
        if self.feature_model is None:
            raise FileNotFoundError('Feature random forest model is not trained yet.')
        missing = [k for k in FEATURE_COLUMNS if k not in feature_row]
        if missing:
            raise ValueError(f'Missing feature values: {missing[:10]}')
        row = pd.DataFrame([{k: feature_row[k] for k in self.selected_features}])
        pred = self.feature_model.predict(row)[0]
        return float(pred)

    def predict_from_formula(self, formula: str, match_level: float = 0.999999, top_k: int = 5) -> FormulaPredictionResult:
        if self.formula_model is None:
            raise FileNotFoundError('Formula random forest model is not trained yet.')

        vec = formula_to_vector(formula, ELEMENTS)
        pred = float(self.formula_model.predict(pd.DataFrame([vec], columns=ELEMENTS))[0])

        input_norm = normalize_vector(vec.to_numpy(dtype=float))
        db_matrix = self.unique_df[ELEMENTS].to_numpy(dtype=float)
        db_norm = np.linalg.norm(db_matrix, axis=1, keepdims=True)
        db_norm[db_norm == 0] = 1.0
        db_matrix_norm = db_matrix / db_norm
        dots = db_matrix_norm @ input_norm.reshape(-1, 1)
        dots = dots.ravel()

        exact_idx = np.where(dots >= match_level)[0]
        exact_matches = self.unique_df.iloc[exact_idx][[FORMULA_COLUMN, TARGET_COLUMN]].copy()
        exact_matches['cosine_similarity'] = dots[exact_idx] if len(exact_idx) else []

        top_idx = np.argsort(dots)[::-1][:top_k]
        similar = self.unique_df.iloc[top_idx][[FORMULA_COLUMN, TARGET_COLUMN]].copy()
        similar['cosine_similarity'] = dots[top_idx]
        return FormulaPredictionResult(
            formula=formula,
            predicted_tc=pred,
            exact_matches=exact_matches,
            similar_materials=similar,
        )
