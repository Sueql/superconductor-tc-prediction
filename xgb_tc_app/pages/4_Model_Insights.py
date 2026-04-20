import streamlit as st
import matplotlib.pyplot as plt
from model_utils import feature_importance_df, get_metrics_snapshot

st.title('Model Insights')
metrics = get_metrics_snapshot()
st.json(metrics)

imp = feature_importance_df()
st.subheader('Top24 feature importance (XGBoost feature model)')
st.dataframe(imp, use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 7))
show = imp.sort_values('importance', ascending=True)
ax.barh(show['feature'], show['importance'])
ax.set_title('Feature importance')
ax.set_xlabel('Importance')
st.pyplot(fig, clear_figure=True)


# st.subheader('How to use these results in presentation/report')
# st.markdown('''
# - Use **Test RMSE / MAE / R²** as the main performance metrics.
# - Use **ACC@±1K / ±5K / ±10K** only as supplementary tolerance-based metrics.
# - Use the feature-importance chart in the **model interpretation** section.
# - Use the formula page as a live demo in the final part of your defense.
# ''')