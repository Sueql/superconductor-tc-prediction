import streamlit as st
from feature_config import TOP24_FEATURES
from model_utils import predict_from_top24, sample_feature_defaults

st.title('Top24 Features → Tc (XGBoost)')
st.write('Enter the 24 selected engineered features and use the XGBoost feature model to predict Tc.')

defaults = sample_feature_defaults()
values = {}
cols = st.columns(2)
for i, feat in enumerate(TOP24_FEATURES):
    col = cols[i % 2]
    values[feat] = col.number_input(feat, value=float(defaults.get(feat, 0.0)), format='%.6f')

if st.button('Predict Tc from top24 features'):
    pred = predict_from_top24(values)
    st.success(f'Predicted Tc = {pred:.3f} K')
