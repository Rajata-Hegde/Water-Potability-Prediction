# Water Potability Classifier

A small machine-learning project that trains a model to classify drinking water as potable or not, and serves interactive Streamlit apps for live predictions.

## What it does
- Trains on `water_potability.csv` with preprocessing (polynomial features + MinMax scaling) and model selection across several classifiers.
- Saves the best model (`l.pkl`) plus transformers (`scaler.pkl`, `polynom.pkl`).
- Provides quick prediction scripts (`1.py`) and two Streamlit UIs (`simpleUI.py`, `heavyUI.py`) for interactive classification.

## Files
- `model.py` — training + EDA; writes `l.pkl`, `scaler.pkl`, `polynom.pkl`.
- `water_potability.csv` — dataset used for training.
- `1.py` — sample one-off prediction.
- `simpleUI.py` — minimal Streamlit form for predictions.
- `heavyUI.py` — richer Streamlit app with history, visualizations, and optional map plotting.
- `data/history.json` — placeholder for persisted history (session history is in-memory by default).

## Models tested
- Extra Trees, RandomForest (with grid search), AdaBoost, Logistic Regression, SVM (with grid search), and XGBoost are trained and compared.
- The script splits first, fits transformers on train only, oversamples only the training split, selects the best model by validation accuracy, then refits on the balanced train and reports held-out test metrics (accuracy, ROC AUC, PR AUC, confusion matrix, classification report). The top model is saved as `l.pkl` and metrics print to the console.

## Quick start
1) Install deps (example):
```bash
pip install  streamlit scikit-learn xgboost seaborn plotly folium streamlit-folium missingno joblib pandas numpy
```
2) Retrain and regenerate artifacts:
```bash
python model.py
```
3) Run a Streamlit app:
```bash
streamlit run simpleUI.py
# or
streamlit run heavyUI.py
```
4) Test a hardcoded sample:
```bash
python 1.py
```

## Notes
- Keep `l.pkl`, `scaler.pkl`, and `polynom.pkl` in the working directory used by the apps.
- The heavy app accepts optional latitude/longitude to drop markers on a Folium map.
- If you update the dataset, rerun `model.py` to refresh the saved artifacts.
