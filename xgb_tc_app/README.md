# XGBoost Interactive Tc App

## Structure

- `Home.py`: home page
- `pages/1_Formula_to_Tc.py`: formula input → Tc prediction
- `pages/2_Top24_Features_to_Tc.py`: top24 features input → Tc prediction
- `pages/3_Batch_Prediction.py`: batch CSV prediction
- `pages/4_Model_Insights.py`: metrics + feature importance

## Required datasets

Put the following files into the `dataset/` folder:

- `dataset/train.csv`
- `dataset/unique_m.csv`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run Home.py
```

## Notes

- Formula page uses `unique_m.csv` and elemental-composition columns.
- Top24 page uses `train.csv` and the selected top24 engineered features.
- The app trains the XGBoost models on startup and caches them.
