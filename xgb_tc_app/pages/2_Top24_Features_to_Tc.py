import streamlit as st

from feature_config import TOP24_FEATURES
from model_utils import predict_from_top24, sample_feature_defaults


def parse_feature_value(raw_value: str, default_value: float, feature_name: str) -> float:
    text = raw_value.strip()
    if not text:
        return float(default_value)
    try:
        return float(text)
    except ValueError:
        st.error(f'Invalid numeric value for {feature_name}: {text}')
        st.stop()


st.title('Top24 Features to Tc (XGBoost)')
st.write('Enter the 24 selected engineered features and use the XGBoost feature model to predict Tc.')

defaults = sample_feature_defaults()
raw_values = {}
cols = st.columns(2)
for i, feat in enumerate(TOP24_FEATURES):
    col = cols[i % 2]
    default_value = float(defaults.get(feat, 0.0))
    raw_values[feat] = col.text_input(
        feat,
        value='',
        placeholder=f'{default_value:.6f}',
        key=f'top24_{feat}',
    )

if st.button('Predict Tc from top24 features'):
    values = {
        feat: parse_feature_value(raw_values[feat], float(defaults.get(feat, 0.0)), feat)
        for feat in TOP24_FEATURES
    }
    pred = predict_from_top24(values)
    st.success(f'Predicted Tc = {pred:.3f} K')
