from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from config import FEATURE_COLUMNS, MODEL_DIR, OUTPUT_DIR
from predictor import SuperconductorPredictor

st.set_page_config(page_title='Superconductor Tc Prediction', layout='wide')
st.title('Superconductor Critical Temperature Prediction')
st.caption('Python adaptation of the original R workflow. Two modes are supported: formula-based prediction and feature-row prediction. The deployed RF model now uses RF-RFE selected top-n features internally.')


@st.cache_resource
def load_predictor() -> SuperconductorPredictor:
    return SuperconductorPredictor()


try:
    predictor = load_predictor()
except Exception as exc:
    st.error(f'Failed to load datasets/models: {exc}')
    st.stop()

with st.sidebar:
    st.header('Project status')
    st.write(f'Model directory: `{MODEL_DIR}`')
    st.write(f'Output directory: `{OUTPUT_DIR}`')
    st.write(f'Selected RF features: **{len(predictor.selected_features)}**')
    if (OUTPUT_DIR / 'metrics_summary.json').exists():
        metrics = json.loads((OUTPUT_DIR / 'metrics_summary.json').read_text(encoding='utf-8'))
        st.json(metrics)
    else:
        st.info('No metrics_summary.json found yet. Run main.py train-all first.')

tab1, tab2 = st.tabs(['Predict from formula', 'Predict from feature row'])

with tab1:
    st.subheader('Formula-based prediction')
    formula = st.text_input('Chemical formula', value='Ba0.2La1.8Cu1O4')
    match_level = st.slider('Near-exact match cosine threshold', min_value=0.90, max_value=1.0, value=0.999999, step=0.000001, format='%.6f')
    if st.button('Predict Tc from formula', type='primary'):
        try:
            result = predictor.predict_from_formula(formula, match_level=match_level)
            st.metric('Predicted Tc (K)', f'{result.predicted_tc:.3f}')
            st.write('### Exact / near-exact matches from unique_m.csv')
            if result.exact_matches.empty:
                st.info('No exact/near-exact match found.')
            else:
                st.dataframe(result.exact_matches, use_container_width=True)
            st.write('### Most similar known superconductors')
            st.dataframe(result.similar_materials, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

with tab2:
    st.subheader('Predict from one feature row')
    st.write('Upload a CSV with one row and the exact 81 feature columns, or edit values manually below. The model will automatically keep only the selected top-n RF-RFE features.')
    upload = st.file_uploader('Upload one-row CSV', type=['csv'])
    feature_values = {}

    if upload is not None:
        row_df = pd.read_csv(upload)
        if len(row_df) != 1:
            st.error('Please upload a CSV with exactly one row.')
        else:
            for col in FEATURE_COLUMNS:
                if col not in row_df.columns:
                    st.error(f'Missing required column: {col}')
                    st.stop()
            feature_values = row_df.iloc[0][FEATURE_COLUMNS].to_dict()
    else:
        defaults = predictor.train_df.iloc[0][FEATURE_COLUMNS].to_dict()
        cols = st.columns(3)
        for i, col in enumerate(FEATURE_COLUMNS):
            with cols[i % 3]:
                feature_values[col] = st.number_input(col, value=float(defaults[col]), format='%.6f')

    st.write('### Features actually used by the deployed RF model')
    st.write(', '.join(predictor.selected_features))

    if st.button('Predict Tc from feature row'):
        try:
            pred = predictor.predict_from_feature_row(feature_values)
            st.metric('Predicted Tc (K)', f'{pred:.3f}')
        except Exception as exc:
            st.error(str(exc))
