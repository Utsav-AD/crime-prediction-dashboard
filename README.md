# London Crime Prediction Dashboard

An interactive web application for analysing and forecasting crime patterns across all 33 London boroughs, built as part of an MSc Data Analytics capstone project at London Metropolitan University.

## Features

- 🗺️ **Interactive choropleth map** of London boroughs with multiple metric views
- 🔍 **Borough deep-dive** with crime category breakdowns and hotspot analysis
- 🔮 **Future predictions** with 95% confidence intervals, risk classification, year-on-year comparison, and downloadable CSV reports
- 🧠 **SHAP explainability** showing why the model makes its predictions
- 📈 **Model performance metrics** with predicted vs actual analysis

## Methodology

- **Data sources:** UK Police API (3.4M+ crime records), London Datastore (socioeconomic indicators), Index of Multiple Deprivation 2019
- **Models:** Ridge Regression, Random Forest, XGBoost
- **Evaluation:** TimeSeriesSplit cross-validation with chronological 80/20 split
- **Best model:** Ridge Regression — Test RMSE = 193.4, R² = 0.976
- **Explainability:** SHAP TreeExplainer applied to Random Forest model

## Tech Stack

- Python 3.11
- Plotly Dash + Dash Bootstrap Components
- scikit-learn, XGBoost, SHAP
- Pandas, NumPy
- Deployed on Render

## Running Locally

```bash
pip install -r requirements.txt
python 10_dashboard.py
```

Then open http://127.0.0.1:8050 in your browser.

## Author

MSc Data Analytics Final Project — London Metropolitan University
