import re
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from config import ELEMENTS

TOKEN_RE = re.compile(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)')


def sanitize_formula(formula: str) -> str:
    formula = formula.strip().replace(' ', '')
    formula = formula.replace('−', '-')
    return formula


def parse_formula(formula: str) -> Dict[str, float]:
    formula = sanitize_formula(formula)
    if not formula:
        raise ValueError('Empty formula.')

    parsed: Dict[str, float] = {}
    pos = 0
    for match in TOKEN_RE.finditer(formula):
        if match.start() != pos:
            raise ValueError(
                f'Unable to parse formula near position {pos}: {formula[pos:match.start()]}'
            )
        symbol, coeff = match.groups()
        value = float(coeff) if coeff else 1.0
        parsed[symbol] = parsed.get(symbol, 0.0) + value
        pos = match.end()

    if pos != len(formula):
        raise ValueError(f'Unable to parse formula tail: {formula[pos:]}')

    unknown = [el for el in parsed if el not in ELEMENTS]
    if unknown:
        raise ValueError(f'Unsupported/unknown elements: {unknown}')
    return parsed


def formula_to_vector(formula: str, elements: Iterable[str] = ELEMENTS) -> pd.Series:
    parsed = parse_formula(formula)
    vec = {el: 0.0 for el in elements}
    vec.update(parsed)
    return pd.Series(vec, dtype=float)


def normalize_vector(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm == 0:
        return x.copy()
    return x / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    denom = np.linalg.norm(a_norm) * np.linalg.norm(b_norm)
    if denom == 0:
        return 0.0
    return float(np.dot(a_norm, b_norm) / denom)
