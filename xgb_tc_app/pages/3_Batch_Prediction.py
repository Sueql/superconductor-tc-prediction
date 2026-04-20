import pandas as pd
import streamlit as st
from model_utils import batch_predict_features, batch_predict_formula
from feature_config import TOP24_FEATURES

st.title('Batch Prediction')
mode = st.radio('Choose batch mode', ['Formula CSV', 'Top24 feature CSV'])
file = st.file_uploader('Upload CSV', type=['csv'])

if file is not None:
    df = pd.read_csv(file)
    try:
        if mode == 'Formula CSV':
            st.info('CSV should contain a column named `material` with chemical formulas.')
            out = batch_predict_formula(df, formula_col='material')
        else:
            st.info('CSV should contain all top24 feature columns.')
            missing = [c for c in TOP24_FEATURES if c not in df.columns]
            if missing:
                st.error(f'Missing columns: {missing}')
                st.stop()
            out = batch_predict_features(df)
        st.dataframe(out.head(50), use_container_width=True)
        st.download_button('Download predictions CSV', out.to_csv(index=False).encode('utf-8'), 'batch_predictions.csv', 'text/csv')
    except Exception as e:
        st.error(str(e))
