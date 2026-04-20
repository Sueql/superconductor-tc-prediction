import streamlit as st
from model_utils import predict_from_formula, get_examples

st.title('Formula → Tc (XGBoost)')
st.write('Input a chemical formula and use the XGBoost formula model to predict Tc.')

examples = get_examples()
# default_formula = examples[0] if examples else 'Ba0.2La1.8CuO4'
default_formula = 'Au1Nb3'
formula = st.text_input('Chemical formula', value=default_formula)


if st.button('Predict Tc from formula'):
    try:
        pred, neighbors = predict_from_formula(formula)
        st.success(f'Predicted Tc = {pred:.3f} K')
        st.subheader('Nearest known materials in composition space')
        st.dataframe(neighbors, use_container_width=True)
        st.caption('These are similar known materials from unique_m.csv; they are useful as qualitative references.')
    except Exception as e:
        st.error(str(e))
