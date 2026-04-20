import re
from collections import defaultdict
from feature_config import ELEMENT_COLUMNS

TOKEN_RE = re.compile(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)')


def parse_formula(formula: str) -> dict:
    formula = formula.strip()
    if not formula:
        raise ValueError('Empty formula.')
    counts = defaultdict(float)
    pos = 0
    for m in TOKEN_RE.finditer(formula):
        if m.start() != pos:
            raise ValueError(f'Cannot parse formula near: {formula[pos:]}')
        elem, qty = m.groups()
        if elem not in ELEMENT_COLUMNS:
            raise ValueError(f'Unsupported element: {elem}')
        counts[elem] += float(qty) if qty else 1.0
        pos = m.end()
    if pos != len(formula):
        raise ValueError(f'Cannot parse formula near: {formula[pos:]}')
    return dict(counts)


def formula_to_vector(formula: str) -> dict:
    counts = parse_formula(formula)
    return {el: counts.get(el, 0.0) for el in ELEMENT_COLUMNS}
