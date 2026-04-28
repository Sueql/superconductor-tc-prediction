from html import escape
from pathlib import Path

import streamlit as st

from model_utils import get_metrics_snapshot


APP_DIR = Path(__file__).resolve().parents[1]
MODEL_INSIGHTS_IMAGE = APP_DIR / 'model_insights.png'

STATIC_FEATURE_ORDER = [
    'wtd_entropy_atomic_mass',
    'wtd_mean_atomic_mass',
    'gmean_Density',
    'wtd_mean_ThermalConductivity',
    'wtd_std_Valence',
    'entropy_Density',
    'gmean_ElectronAffinity',
    'mean_Density',
    'std_Density',
    'wtd_range_atomic_mass',
    'wtd_entropy_FusionHeat',
    'wtd_std_ThermalConductivity',
    'range_atomic_radius',
    'wtd_std_ElectronAffinity',
    'wtd_mean_Valence',
    'wtd_entropy_ThermalConductivity',
    'std_atomic_mass',
    'wtd_gmean_ThermalConductivity',
    'wtd_gmean_Valence',
    'range_ThermalConductivity',
]


def render_feature_table(features: list[str]) -> str:
    rows = '\n'.join(
        f'<tr><td>{idx}</td><td>{escape(feature)}</td></tr>'
        for idx, feature in enumerate(features, start=1)
    )
    return f'''
<style>
  .feature-table-wrapper {{
    width: 100%;
    overflow-x: auto;
  }}
  .feature-table {{
    width: 100%;
    border-collapse: collapse;
    table-layout: auto;
  }}
  .feature-table th,
  .feature-table td {{
    border: 1px solid rgba(49, 51, 63, 0.2);
    padding: 0.55rem 0.75rem;
    vertical-align: middle;
    word-break: break-word;
  }}
  .feature-table th {{
    background: rgba(49, 51, 63, 0.06);
    font-weight: 600;
  }}
  .feature-table th:first-child,
  .feature-table td:first-child {{
    text-align: center;
    white-space: nowrap;
  }}
  .feature-table th:nth-child(2),
  .feature-table td:nth-child(2) {{
    text-align: center;
  }}
</style>
<div class="feature-table-wrapper">
  <table class="feature-table">
    <thead>
      <tr>
        <th>No.</th>
        <th>Feature</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</div>
'''


st.title('Model Insights')

metrics = get_metrics_snapshot()
st.json(metrics)

st.subheader('Top-20 features from model insights')
st.markdown(render_feature_table(STATIC_FEATURE_ORDER), unsafe_allow_html=True)

st.subheader('Model insights visualization')
if MODEL_INSIGHTS_IMAGE.exists():
    st.image(str(MODEL_INSIGHTS_IMAGE), use_container_width=True)
else:
    st.error(f'Missing image file: {MODEL_INSIGHTS_IMAGE}')
