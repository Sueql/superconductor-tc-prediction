import streamlit as st

from model_utils import predict_from_formula


DEFAULT_FORMULA = 'Au1Nb3'


st.title('Formula to Tc (XGBoost)')
st.write('Input a chemical formula and use the XGBoost formula model to predict Tc.')

formula = st.text_input('Chemical formula', value='', placeholder=DEFAULT_FORMULA)

if st.button('Predict Tc from formula'):
    try:
        formula_to_predict = formula.strip() or DEFAULT_FORMULA
        pred, neighbors = predict_from_formula(formula_to_predict)
        st.success(f'Predicted Tc = {pred:.3f} K')
        st.subheader('Nearest known materials in composition space')
        st.dataframe(neighbors, use_container_width=True)
        st.caption('These are similar known materials from unique_m.csv; they are useful as qualitative references.')
    except Exception as e:
        st.error(str(e))
