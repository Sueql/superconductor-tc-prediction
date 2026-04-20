import streamlit as st
from model_utils import get_metrics_snapshot, get_examples

st.set_page_config(page_title='Tc Prediction App', page_icon='🧪', layout='wide')

st.title('Superconductor Tc Prediction App')
st.write('This app provides two XGBoost-based interactive predictors: formula → Tc and top24 features → Tc.')

metrics = get_metrics_snapshot()
cols = st.columns(5)
cols[0].metric('Best model', metrics['best_model'])
cols[1].metric('Test RMSE', f"{metrics['test_RMSE']:.3f}")
cols[2].metric('Test MAE', f"{metrics['test_MAE']:.3f}")
cols[3].metric('Test R²', f"{metrics['test_R2']:.3f}")
cols[4].metric('ACC@±10K', f"{metrics['ACC@±10K']:.3f}")

st.subheader('Pages')
st.markdown('''
- **Formula → Tc**: type a chemical formula such as `MgB2` or `Au1Nb3`.
- **Top24 Features → Tc**: enter the 24 engineered descriptors and predict Tc.
- **Batch Prediction**: upload CSV for batch inference.
- **Model Insights**: see feature importance and project summary.
''')

st.subheader('Quick Examples')
st.write(get_examples())
### examples = get_examples()
### st.write(examples)

#st.info("You're all set to start exploring!")
st.info("Everything is ready! Enter a chemical formula, input feature values, or upload a CSV file to start predicting superconducting critical temperatures (Tc) right away.")