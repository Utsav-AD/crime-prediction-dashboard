"""
Phase 5b — Interactive Plotly Dash Dashboard (Enhanced)
========================================================
A web app for exploring crime predictions, trends, and SHAP explanations.

The Future Predictions tab now includes:
  • Confidence intervals (95% uncertainty bands)
  • Crime category breakdown for forecasted months
  • Year-on-year comparison vs same month last year
  • Multi-borough comparison (up to 5 boroughs at once)
  • Risk classification (low / medium / high / very high)
  • Downloadable forecast report (CSV)

Usage:
  python 10_dashboard.py
  Then open http://127.0.0.1:8050 in your browser.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import warnings
warnings.filterwarnings("ignore")

import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/random_forest.pkl"
DATA_PATH = "data/processed/modelling_dataset.csv"
GEOJSON_PATH = "data/raw/london_boroughs.geojson"
RAW_DATA_PATH = "data/raw/crime_data_london_raw.csv"

TREE_FEATURES = [
    "year", "month_num", "quarter", "month_sin", "month_cos", "season",
    "crime_lag_1", "crime_lag_3", "crime_lag_12",
    "crime_rolling_3", "crime_rolling_6",
    "imd_score", "income_deprivation_score", "employment_deprivation_score",
    "crime_deprivation_score",
    "population_2023", "population_density_per_km2",
    "claimant_count_rate_2023", "median_annual_earnings_2023",
    "median_house_price_2023", "overcrowding_rate",
]

SEASON_MAP = {12: "winter", 1: "winter", 2: "winter",
              3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer",
              9: "autumn", 10: "autumn", 11: "autumn"}

RISK_LEVELS = {
    "very_high": {"label": "Very High", "color": "#8B0000", "threshold": 7000},
    "high":      {"label": "High",      "color": "#E63946", "threshold": 4000},
    "medium":    {"label": "Medium",    "color": "#F77F00", "threshold": 2500},
    "low":       {"label": "Low",       "color": "#52B788", "threshold": 0},
}

def classify_risk(value):
    if value >= RISK_LEVELS["very_high"]["threshold"]: return "very_high"
    elif value >= RISK_LEVELS["high"]["threshold"]:    return "high"
    elif value >= RISK_LEVELS["medium"]["threshold"]:  return "medium"
    return "low"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Loading dashboard resources...")
print("=" * 60)

print("\n[1/5] Loading model...")
pipeline = joblib.load(MODEL_PATH)
preprocessor = pipeline.named_steps["prep"]
rf_model = pipeline.named_steps["model"]

print("[2/5] Loading datasets...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["crime_lag_1", "crime_lag_3", "crime_lag_12",
                        "crime_rolling_3", "crime_rolling_6"]).reset_index(drop=True)
raw_df = pd.read_csv(RAW_DATA_PATH, usecols=["borough", "month", "category", "street"])
with open(GEOJSON_PATH) as f:
    geojson = json.load(f)
for feat in geojson["features"]:
    feat["properties"]["name"] = feat["properties"].get("name") or feat["properties"].get("NAME") or ""

print("[3/5] Setting up SHAP explainer...")
num_features = [f for f in TREE_FEATURES if f != "season"]
cat_encoded_features = [f"season_{s}" for s in ["spring", "summer", "winter"]]
all_feature_names = num_features + cat_encoded_features
explainer = shap.TreeExplainer(rf_model)

print("[4/5] Computing predictions...")
boroughs = sorted(df["borough"].unique())
months = sorted(df["month"].unique())
df["predicted"] = pipeline.predict(df[TREE_FEATURES])

print("[5/5] Computing residuals for confidence intervals...")
test_start = sorted(df["date"].unique())[int(len(df["date"].unique()) * 0.8)]
test_data = df[df["date"] >= test_start]
RESIDUALS = (test_data["crime_count"] - test_data["predicted"]).values
RESIDUAL_LOWER = float(np.quantile(RESIDUALS, 0.025))
RESIDUAL_UPPER = float(np.quantile(RESIDUALS, 0.975))

print(f"\n✓ Loaded: {len(df)} rows, {len(boroughs)} boroughs, {len(months)} months")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_row(history, target_date, borough):
    h = history[history["borough"] == borough].sort_values("date").reset_index(drop=True)
    socio_cols = ["imd_score", "income_deprivation_score", "employment_deprivation_score",
                  "crime_deprivation_score", "population_2023", "population_density_per_km2",
                  "claimant_count_rate_2023", "median_annual_earnings_2023",
                  "median_house_price_2023", "overcrowding_rate"]
    socio = h.iloc[-1][socio_cols].to_dict()
    lag_1  = h.iloc[-1]["crime_count"]
    lag_3  = h.iloc[-3]["crime_count"]  if len(h) >= 3  else lag_1
    lag_12 = h.iloc[-12]["crime_count"] if len(h) >= 12 else lag_1
    rolling_3 = h.iloc[-3:]["crime_count"].mean() if len(h) >= 3 else lag_1
    rolling_6 = h.iloc[-6:]["crime_count"].mean() if len(h) >= 6 else lag_1
    month_num = target_date.month
    return pd.DataFrame([{
        "year": target_date.year, "month_num": month_num,
        "quarter": (month_num - 1) // 3 + 1,
        "month_sin": np.sin(2 * np.pi * month_num / 12),
        "month_cos": np.cos(2 * np.pi * month_num / 12),
        "season": SEASON_MAP[month_num],
        "crime_lag_1": lag_1, "crime_lag_3": lag_3, "crime_lag_12": lag_12,
        "crime_rolling_3": rolling_3, "crime_rolling_6": rolling_6, **socio,
    }])

def forecast_recursive(history, borough, num_months):
    """Forecast with widening confidence intervals over time."""
    h = history.copy()
    forecasts = []
    last_date = h[h["borough"] == borough]["date"].max()
    for i in range(num_months):
        target_date = last_date + pd.DateOffset(months=1)
        X = build_feature_row(h, target_date, borough)
        pred = float(pipeline.predict(X[TREE_FEATURES])[0])
        horizon_factor = np.sqrt(i + 1)
        lower = pred + RESIDUAL_LOWER * horizon_factor
        upper = pred + RESIDUAL_UPPER * horizon_factor
        forecasts.append({
            "date": target_date, "month": target_date.strftime("%Y-%m"),
            "prediction": pred, "lower_95": max(0, lower), "upper_95": upper,
            "borough": borough, "horizon": i + 1, "risk": classify_risk(pred),
        })
        new_row = X.iloc[0].to_dict()
        new_row["borough"], new_row["date"] = borough, target_date
        new_row["month"], new_row["crime_count"] = target_date.strftime("%Y-%m"), pred
        h = pd.concat([h, pd.DataFrame([new_row])], ignore_index=True)
        last_date = target_date
    return pd.DataFrame(forecasts)

def get_category_breakdown(borough, predicted_total):
    bdf = raw_df[raw_df["borough"] == borough]
    last_12 = sorted(bdf["month"].unique())[-12:]
    recent = bdf[bdf["month"].isin(last_12)]
    proportions = recent["category"].value_counts(normalize=True)
    return (proportions * predicted_total).round().astype(int)

def get_yoy_comparison(borough, target_date):
    last_year = target_date - pd.DateOffset(years=1)
    yoy = df[(df["borough"] == borough) & (df["date"] == last_year)]
    return float(yoy["crime_count"].iloc[0]) if len(yoy) else None

# ─────────────────────────────────────────────────────────────────────────────
# DASH APP
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    title="London Crime Prediction Dashboard",
    suppress_callback_exceptions=True,
)
PRIMARY = "#1F3864"
ACCENT = "#E63946"

app.layout = dbc.Container([
    dbc.Row([dbc.Col([
        html.Div([
            html.H1([html.I(className="fas fa-shield-alt me-3", style={"color": PRIMARY}),
                     "London Crime Prediction Dashboard"],
                    style={"color": PRIMARY, "fontWeight": "bold"}),
            html.P([
                "MSc Data Analytics Capstone Project · ",
                html.Span("Random Forest + SHAP Explainability",
                          style={"color": "#666", "fontStyle": "italic"})
            ], className="lead text-muted"),
        ], className="py-4 border-bottom")
    ])]),

    dbc.Tabs([
        # TAB 1: MAP
        dbc.Tab(label="🗺️ Map & Overview", tab_id="tab-map", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Select month:", className="fw-bold"),
                    dcc.Dropdown(id="map-month",
                                 options=[{"label": m, "value": m} for m in months],
                                 value=months[-1], clearable=False),
                ], md=4),
                dbc.Col([
                    html.Label("Display metric:", className="fw-bold"),
                    dcc.RadioItems(id="map-metric",
                                   options=[{"label": " Total crime count", "value": "crime_count"},
                                            {"label": " Crime rate per 1,000", "value": "crime_rate_per_1000"},
                                            {"label": " Predicted vs actual error", "value": "error"}],
                                   value="crime_count",
                                   labelStyle={"display": "inline-block", "marginRight": "20px"}),
                ], md=8),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([dcc.Loading(dcc.Graph(id="choropleth-map", style={"height": "600px"}))], md=8),
                dbc.Col([html.Div(id="map-summary-cards")], md=4),
            ]),
            html.Hr(),
            dbc.Row([dbc.Col([
                html.H4("📊 Borough rankings", className="mt-3"),
                dcc.Loading(dcc.Graph(id="borough-ranking-chart"))
            ])]),
        ]),

        # TAB 2: BOROUGH DEEP DIVE
        dbc.Tab(label="🔍 Borough Deep Dive", tab_id="tab-borough", children=[
            html.Br(),
            dbc.Row([dbc.Col([
                html.Label("Select borough:", className="fw-bold"),
                dcc.Dropdown(id="borough-select",
                             options=[{"label": b, "value": b} for b in boroughs],
                             value="Westminster", clearable=False),
            ], md=12)], className="mb-3"),
            dbc.Row([dbc.Col([dcc.Loading(dcc.Graph(id="borough-timeseries", style={"height": "450px"}))])]),
            dbc.Row([
                dbc.Col([dcc.Loading(dcc.Graph(id="category-pie"))], md=6),
                dbc.Col([dcc.Loading(dcc.Graph(id="hotspot-bar"))], md=6),
            ]),
        ]),

        # TAB 3: FUTURE PREDICTIONS (ENHANCED)
        dbc.Tab(label="🔮 Future Predictions", tab_id="tab-future", children=[
            html.Br(),
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                html.B("How this works: "),
                "Forecasts use recursive prediction, where each month's forecast feeds into the next. ",
                "Confidence intervals widen as we forecast further out, reflecting compound uncertainty."
            ], color="info"),

            dbc.Row([
                dbc.Col([
                    html.Label("Borough(s) — pick up to 5 to compare:", className="fw-bold"),
                    dcc.Dropdown(id="fc-boroughs",
                                 options=[{"label": b, "value": b} for b in boroughs],
                                 value=["Westminster", "Camden", "Bexley"], multi=True),
                ], md=6),
                dbc.Col([
                    html.Label("Forecast horizon (months):", className="fw-bold"),
                    dcc.Slider(id="fc-horizon", min=1, max=12, step=1, value=6,
                               marks={i: str(i) for i in range(1, 13)}),
                ], md=6),
            ], className="mb-4"),

            dbc.Row([dbc.Col([
                html.Div([
                    html.Span("Risk levels: ", className="fw-bold me-3"),
                    *[html.Span([
                        html.Span("●", style={"color": v["color"], "fontSize": "1.4em",
                                              "marginRight": "5px"}),
                        f"{v['label']} ",
                        html.Span(f"(≥{v['threshold']:,})" if v['threshold'] > 0 else "(<2,500)",
                                  style={"color": "#888", "fontSize": "0.85em"}),
                        html.Span(" · ", className="text-muted mx-2") if k != "low" else "",
                    ]) for k, v in RISK_LEVELS.items()]
                ], className="text-center py-2 px-3 bg-light rounded mb-3")
            ])]),

            dbc.Row([dbc.Col([
                dcc.Loading(dcc.Graph(id="fc-comparison-chart", style={"height": "500px"}))
            ])]),

            html.H4("🚦 Risk timeline", className="mt-4 mb-3"),
            dcc.Loading(html.Div(id="fc-risk-timeline")),

            html.Hr(),
            html.H4("📅 Year-on-year comparison", className="mt-3 mb-3"),
            dbc.Row([dbc.Col([dcc.Loading(dcc.Graph(id="fc-yoy-chart"))])]),

            html.Hr(),
            html.H4("🥧 Predicted crime category breakdown", className="mt-3 mb-3"),
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "Categories estimated using each borough's last 12 months of recorded patterns. ",
                "Shown for the first selected borough."
            ], color="light"),
            dbc.Row([
                dbc.Col([dcc.Loading(dcc.Graph(id="fc-category-chart"))], md=7),
                dbc.Col([dcc.Loading(html.Div(id="fc-category-table"))], md=5),
            ]),

            html.Hr(),
            html.Div([
                html.H4("📄 Download forecast report", className="mt-3"),
                html.P("Export a complete CSV with forecasts, confidence intervals, "
                       "risk classifications, and year-on-year comparisons "
                       "for all selected boroughs.", className="text-muted"),
                dbc.Button([html.I(className="fas fa-download me-2"),
                            "Download Forecast Report (CSV)"],
                           id="fc-download-btn", color="primary", size="lg"),
                dcc.Download(id="fc-download"),
            ], className="text-center py-4"),
        ]),

        # TAB 4: SHAP
        dbc.Tab(label="🧠 SHAP Explainability", tab_id="tab-shap", children=[
            html.Br(),
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "SHAP values explain why the model made a specific prediction. ",
                "Red bars push the prediction higher; blue bars push it lower."
            ], color="info"),
            dbc.Row([
                dbc.Col([
                    html.Label("Borough:", className="fw-bold"),
                    dcc.Dropdown(id="shap-borough",
                                 options=[{"label": b, "value": b} for b in boroughs],
                                 value="Westminster", clearable=False),
                ], md=6),
                dbc.Col([
                    html.Label("Month:", className="fw-bold"),
                    dcc.Dropdown(id="shap-month",
                                 options=[{"label": m, "value": m} for m in months],
                                 value=months[-1], clearable=False),
                ], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([dcc.Loading(dcc.Graph(id="shap-waterfall", style={"height": "550px"}))], md=8),
                dbc.Col([html.Div(id="shap-summary-cards")], md=4),
            ]),
        ]),

        # TAB 5: PERFORMANCE
        dbc.Tab(label="📈 Model Performance", tab_id="tab-perf", children=[
            html.Br(),
            html.Div(id="performance-content"),
        ]),
    ], id="tabs", active_tab="tab-map"),

    html.Hr(),
    html.P([
        "Built with Plotly Dash · Data: UK Police API + London Datastore · ",
        html.Span("MSc Data Analytics, London Metropolitan University",
                  style={"fontStyle": "italic"})
    ], className="text-center text-muted small py-3"),
], fluid=True)


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — TAB 1
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(Output("choropleth-map", "figure"), Output("map-summary-cards", "children"),
              Input("map-month", "value"), Input("map-metric", "value"))
def update_map(selected_month, metric):
    sub = df[df["month"] == selected_month].copy()
    sub["error"] = (sub["predicted"] - sub["crime_count"]).abs()
    label_map = {"crime_count": "Crime count",
                 "crime_rate_per_1000": "Crime rate (per 1,000)",
                 "error": "Prediction error"}
    fig = px.choropleth_mapbox(
        sub, geojson=geojson, locations="borough",
        featureidkey="properties.name", color=metric,
        color_continuous_scale="Reds" if metric != "error" else "Oranges",
        center={"lat": 51.5074, "lon": -0.1278}, zoom=8.5,
        mapbox_style="carto-positron",
        hover_data={"borough": True, "crime_count": ":,",
                    "predicted": ":,.0f", "crime_rate_per_1000": ":.2f"},
        labels={metric: label_map[metric]},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))

    total = sub["crime_count"].sum()
    top = sub.nlargest(1, "crime_count").iloc[0]
    avg_error_pct = (sub["error"] / sub["crime_count"]).mean() * 100
    cards = dbc.Stack([
        dbc.Card(dbc.CardBody([
            html.H6("Total crimes", className="text-muted mb-1"),
            html.H3(f"{total:,.0f}", style={"color": PRIMARY, "margin": 0}),
            html.Small(f"across all 33 boroughs in {selected_month}", className="text-muted"),
        ]), className="shadow-sm"),
        dbc.Card(dbc.CardBody([
            html.H6("Highest borough", className="text-muted mb-1"),
            html.H3(top["borough"], style={"color": ACCENT, "margin": 0, "fontSize": "1.6rem"}),
            html.Small(f"{top['crime_count']:,} crimes", className="text-muted"),
        ]), className="shadow-sm"),
        dbc.Card(dbc.CardBody([
            html.H6("Avg model error", className="text-muted mb-1"),
            html.H3(f"{avg_error_pct:.1f}%", style={"color": "#52B788", "margin": 0}),
            html.Small("mean absolute % error this month", className="text-muted"),
        ]), className="shadow-sm"),
    ], gap=3)
    return fig, cards


@app.callback(Output("borough-ranking-chart", "figure"), Input("map-month", "value"))
def update_rankings(selected_month):
    sub = df[df["month"] == selected_month].sort_values("crime_count", ascending=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Crime count by borough", "Crime rate per 1,000"))
    fig.add_trace(go.Bar(x=sub["crime_count"], y=sub["borough"], orientation="h",
                         marker_color=PRIMARY,
                         hovertemplate="<b>%{y}</b><br>%{x:,.0f} crimes<extra></extra>"), 1, 1)
    sub_rate = sub.sort_values("crime_rate_per_1000", ascending=True)
    fig.add_trace(go.Bar(x=sub_rate["crime_rate_per_1000"], y=sub_rate["borough"],
                         orientation="h", marker_color=ACCENT,
                         hovertemplate="<b>%{y}</b><br>%{x:.2f} per 1,000<extra></extra>"), 1, 2)
    fig.update_layout(showlegend=False, height=750, margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — TAB 2
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(Output("borough-timeseries", "figure"), Input("borough-select", "value"))
def update_borough_timeseries(borough):
    bdf = df[df["borough"] == borough].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bdf["date"], y=bdf["crime_count"],
                             mode="lines+markers", name="Actual",
                             line=dict(color=PRIMARY, width=2.5), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=bdf["date"], y=bdf["predicted"],
                             mode="lines", name="Model fit",
                             line=dict(color=ACCENT, width=1.5, dash="dot")))
    fig.update_layout(
        title=f"<b>{borough}</b> — historical crime",
        xaxis_title="Month", yaxis_title="Crime count",
        hovermode="x unified", margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
    )
    return fig


@app.callback(Output("category-pie", "figure"), Output("hotspot-bar", "figure"),
              Input("borough-select", "value"))
def update_borough_breakdown(borough):
    rdf = raw_df[raw_df["borough"] == borough]
    last_12 = sorted(rdf["month"].unique())[-12:]
    rdf = rdf[rdf["month"].isin(last_12)]

    cat_counts = rdf["category"].value_counts().head(8)
    pie = px.pie(names=cat_counts.index.str.replace("-", " ").str.title(),
                 values=cat_counts.values,
                 title=f"<b>Top crime categories in {borough}</b><br><sub>last 12 months</sub>",
                 color_discrete_sequence=px.colors.sequential.RdBu_r)
    pie.update_traces(textinfo="percent+label", textposition="inside")
    pie.update_layout(margin=dict(l=0, r=0, t=70, b=0), showlegend=False)

    streets = rdf["street"].dropna()
    streets = streets[streets.str.strip() != ""]
    top_streets = streets.value_counts().head(10).iloc[::-1]
    bar = go.Figure(go.Bar(x=top_streets.values, y=top_streets.index, orientation="h",
                           marker_color=PRIMARY,
                           hovertemplate="<b>%{y}</b><br>%{x:,} crimes<extra></extra>"))
    bar.update_layout(title=f"<b>Top hotspot streets in {borough}</b><br><sub>last 12 months</sub>",
                      margin=dict(l=0, r=0, t=70, b=0), xaxis_title="Crime count")
    return pie, bar


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — TAB 3: FUTURE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("fc-comparison-chart", "figure"),
    Output("fc-risk-timeline", "children"),
    Output("fc-yoy-chart", "figure"),
    Output("fc-category-chart", "figure"),
    Output("fc-category-table", "children"),
    Input("fc-boroughs", "value"),
    Input("fc-horizon", "value"),
)
def update_forecast(selected_boroughs, horizon):
    if not selected_boroughs:
        return (go.Figure(), html.Div("Select at least one borough."),
                go.Figure(), go.Figure(), html.Div())
    selected_boroughs = selected_boroughs[:5]

    all_forecasts = [forecast_recursive(df, b, horizon) for b in selected_boroughs]
    fc_df = pd.concat(all_forecasts, ignore_index=True)

    palette = ["#1F3864", "#E63946", "#52B788", "#F77F00", "#7209B7"]

    # 1. Comparison chart with confidence bands
    comp_fig = go.Figure()
    for i, b in enumerate(selected_boroughs):
        bfc = fc_df[fc_df["borough"] == b].copy()
        hist = df[df["borough"] == b].sort_values("date").tail(6)
        color = palette[i % len(palette)]

        comp_fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["crime_count"],
            mode="lines+markers", name=f"{b} (actual)",
            line=dict(color=color, width=2),
            marker=dict(size=6), legendgroup=b,
        ))

        bridge_x = [hist.iloc[-1]["date"], bfc.iloc[0]["date"]]
        bridge_y = [hist.iloc[-1]["crime_count"], bfc.iloc[0]["prediction"]]
        comp_fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y, mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            showlegend=False, legendgroup=b, hoverinfo="skip",
        ))

        comp_fig.add_trace(go.Scatter(
            x=bfc["date"].tolist() + bfc["date"].tolist()[::-1],
            y=bfc["upper_95"].tolist() + bfc["lower_95"].tolist()[::-1],
            fill="toself", fillcolor=color, opacity=0.15,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{b} 95% CI", legendgroup=b, showlegend=False,
            hoverinfo="skip",
        ))

        comp_fig.add_trace(go.Scatter(
            x=bfc["date"], y=bfc["prediction"],
            mode="lines+markers", name=f"{b} (forecast)",
            line=dict(color=color, width=2.5, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
            legendgroup=b, customdata=bfc[["lower_95", "upper_95", "risk"]],
            hovertemplate="<b>" + b + "</b><br>%{x|%Y-%m}<br>"
                          "Predicted: %{y:,.0f}<br>"
                          "95% CI: %{customdata[0]:,.0f}–%{customdata[1]:,.0f}<br>"
                          "Risk: %{customdata[2]}<extra></extra>",
        ))

    comp_fig.update_layout(
        title="<b>Forecast comparison with 95% confidence intervals</b>",
        xaxis_title="Month", yaxis_title="Crime count",
        hovermode="x unified", margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )

    # 2. Risk timeline
    risk_rows = []
    for b in selected_boroughs:
        bfc = fc_df[fc_df["borough"] == b]
        cards = []
        for _, row in bfc.iterrows():
            r = RISK_LEVELS[row["risk"]]
            cards.append(dbc.Card(dbc.CardBody([
                html.Small(row["month"], className="text-muted d-block"),
                html.Span("●", style={"color": r["color"], "fontSize": "1.5em",
                                      "verticalAlign": "middle"}),
                html.Span(f" {r['label']}", style={"fontWeight": "600",
                                                   "verticalAlign": "middle"}),
                html.Br(),
                html.Span(f"{row['prediction']:,.0f}",
                          style={"fontWeight": "bold", "fontSize": "1.1em"}),
            ], className="text-center py-2 px-2"),
            style={"borderLeft": f"4px solid {r['color']}", "minWidth": "110px"},
            className="shadow-sm"))
        risk_rows.append(dbc.Row([
            dbc.Col(html.H6(b, className="fw-bold pt-2"), md=2),
            dbc.Col(dbc.Stack(cards, direction="horizontal", gap=2,
                              style={"overflowX": "auto"}), md=10),
        ], className="mb-3"))
    risk_timeline = html.Div(risk_rows)

    # 3. Year-on-year
    yoy_data = []
    for b in selected_boroughs:
        bfc = fc_df[fc_df["borough"] == b]
        for _, row in bfc.iterrows():
            yoy_actual = get_yoy_comparison(b, row["date"])
            yoy_data.append({
                "borough": b, "month": row["month"],
                "predicted": row["prediction"],
                "yoy_actual": yoy_actual if yoy_actual else 0,
                "change_pct": ((row["prediction"] - yoy_actual) / yoy_actual * 100)
                              if yoy_actual else 0,
            })
    yoy_df_local = pd.DataFrame(yoy_data)

    yoy_fig = go.Figure()
    for i, b in enumerate(selected_boroughs):
        sub = yoy_df_local[yoy_df_local["borough"] == b]
        color = palette[i % len(palette)]
        yoy_fig.add_trace(go.Bar(
            x=sub["month"], y=sub["change_pct"], name=b, marker_color=color,
            customdata=sub[["predicted", "yoy_actual"]],
            hovertemplate="<b>" + b + "</b><br>%{x}<br>"
                          "Predicted: %{customdata[0]:,.0f}<br>"
                          "Same month last year: %{customdata[1]:,.0f}<br>"
                          "Change: %{y:+.1f}%<extra></extra>",
        ))
    yoy_fig.add_hline(y=0, line_color="black", line_width=1)
    yoy_fig.update_layout(title="<b>Forecast vs same month last year</b>",
                          xaxis_title="Forecast month",
                          yaxis_title="% change vs last year",
                          barmode="group", margin=dict(l=0, r=0, t=50, b=0))

    # 4. Category breakdown for first borough
    primary = selected_boroughs[0]
    primary_total = fc_df[fc_df["borough"] == primary]["prediction"].sum()
    cat_breakdown = get_category_breakdown(primary, primary_total)
    cat_top = cat_breakdown.head(8)

    cat_fig = px.bar(
        x=cat_top.values, y=[c.replace("-", " ").title() for c in cat_top.index],
        orientation="h",
        title=f"<b>Predicted crime categories — {primary}</b><br>"
              f"<sub>Total over {horizon} months: {primary_total:,.0f}</sub>",
        color=cat_top.values, color_continuous_scale="Reds",
        labels={"x": "Predicted crimes", "y": ""},
    )
    cat_fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=70, b=0))

    cat_table = dbc.Table([
        html.Thead(html.Tr([html.Th("Category"), html.Th("Predicted"), html.Th("%")])),
        html.Tbody([
            html.Tr([
                html.Td(c.replace("-", " ").title()),
                html.Td(f"{int(v):,}"),
                html.Td(f"{v / primary_total * 100:.1f}%"),
            ]) for c, v in cat_top.items()
        ])
    ], bordered=True, hover=True, striped=True, size="sm", className="mt-5")

    return comp_fig, risk_timeline, yoy_fig, cat_fig, cat_table


@app.callback(Output("fc-download", "data"), Input("fc-download-btn", "n_clicks"),
              State("fc-boroughs", "value"), State("fc-horizon", "value"),
              prevent_initial_call=True)
def download_report(n_clicks, selected_boroughs, horizon):
    if not selected_boroughs:
        return no_update
    rows = []
    for b in selected_boroughs:
        fc = forecast_recursive(df, b, horizon)
        cat_breakdown = get_category_breakdown(b, fc["prediction"].sum())
        for _, row in fc.iterrows():
            yoy = get_yoy_comparison(b, row["date"])
            rows.append({
                "Borough": b, "Forecast Month": row["month"],
                "Horizon (months ahead)": row["horizon"],
                "Predicted Crime Count": round(row["prediction"]),
                "95% CI Lower": round(row["lower_95"]),
                "95% CI Upper": round(row["upper_95"]),
                "Risk Level": RISK_LEVELS[row["risk"]]["label"],
                "Same Month Last Year": round(yoy) if yoy else "N/A",
                "YoY % Change": f"{((row['prediction'] - yoy) / yoy * 100):+.1f}%" if yoy else "N/A",
                "Top Category": cat_breakdown.index[0].replace("-", " ").title() if len(cat_breakdown) else "N/A",
                "Top Category Count": int(cat_breakdown.iloc[0]) if len(cat_breakdown) else 0,
            })
    report_df = pd.DataFrame(rows)
    return dcc.send_data_frame(report_df.to_csv, "crime_forecast_report.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — TAB 4 (SHAP)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(Output("shap-waterfall", "figure"), Output("shap-summary-cards", "children"),
              Input("shap-borough", "value"), Input("shap-month", "value"))
def update_shap(borough, month):
    row = df[(df["borough"] == borough) & (df["month"] == month)]
    if len(row) == 0:
        return go.Figure(), html.Div("No data available for this combination.")
    X_row = row[TREE_FEATURES]
    X_transformed = preprocessor.transform(X_row)
    X_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    shap_values = explainer.shap_values(X_df)[0]
    base = float(explainer.expected_value)
    pred = base + float(shap_values.sum())
    actual = float(row["crime_count"].iloc[0])
    contrib_df = pd.DataFrame({"feature": all_feature_names, "shap_value": shap_values})
    contrib_df["abs_shap"] = contrib_df["shap_value"].abs()
    contrib_df = contrib_df.nlargest(12, "abs_shap").sort_values("shap_value")
    colors = [ACCENT if v > 0 else PRIMARY for v in contrib_df["shap_value"]]
    fig = go.Figure(go.Bar(x=contrib_df["shap_value"], y=contrib_df["feature"],
                           orientation="h", marker_color=colors,
                           hovertemplate="<b>%{y}</b><br>SHAP impact: %{x:+.0f}<extra></extra>"))
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
    fig.update_layout(
        title=f"<b>Why the model predicts {pred:,.0f} crimes for {borough}, {month}</b><br>"
              f"<sub>Base: {base:,.0f}  →  Predicted: {pred:,.0f}  (Actual: {actual:,.0f})</sub>",
        xaxis_title="SHAP value (impact on prediction)",
        margin=dict(l=0, r=0, t=70, b=0),
    )
    error = abs(actual - pred)
    error_pct = error / actual * 100
    cards = dbc.Stack([
        dbc.Card(dbc.CardBody([
            html.H6("Baseline", className="text-muted mb-1"),
            html.H4(f"{base:,.0f}", style={"color": "#666", "margin": 0}),
            html.Small("avg prediction across dataset", className="text-muted"),
        ]), className="shadow-sm"),
        dbc.Card(dbc.CardBody([
            html.H6("Predicted", className="text-muted mb-1"),
            html.H4(f"{pred:,.0f}", style={"color": PRIMARY, "margin": 0}),
            html.Small("model output for this borough/month", className="text-muted"),
        ]), className="shadow-sm"),
        dbc.Card(dbc.CardBody([
            html.H6("Actual", className="text-muted mb-1"),
            html.H4(f"{actual:,.0f}", style={"color": ACCENT, "margin": 0}),
            html.Small("reported crimes (UK Police data)", className="text-muted"),
        ]), className="shadow-sm"),
        dbc.Card(dbc.CardBody([
            html.H6("Model error", className="text-muted mb-1"),
            html.H4(f"{error:,.0f}", style={"color": "#52B788", "margin": 0}),
            html.Small(f"{error_pct:.1f}% off", className="text-muted"),
        ]), className="shadow-sm"),
    ], gap=3)
    return fig, cards


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS — TAB 5 (PERFORMANCE)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(Output("performance-content", "children"), Input("tabs", "active_tab"))
def render_performance(active_tab):
    if active_tab != "tab-perf":
        return no_update
    test_start_local = sorted(df["date"].unique())[int(len(df["date"].unique()) * 0.8)]
    test = df[df["date"] >= test_start_local]
    rmse = np.sqrt(((test["crime_count"] - test["predicted"]) ** 2).mean())
    mae = (test["crime_count"] - test["predicted"]).abs().mean()
    r2 = 1 - ((test["crime_count"] - test["predicted"]) ** 2).sum() / \
            ((test["crime_count"] - test["crime_count"].mean()) ** 2).sum()

    scatter = go.Figure()
    scatter.add_trace(go.Scatter(x=test["crime_count"], y=test["predicted"], mode="markers",
                                  marker=dict(size=8, color=PRIMARY, opacity=0.6),
                                  text=test["borough"] + " — " + test["month"],
                                  hovertemplate="<b>%{text}</b><br>Actual: %{x:,}<br>Predicted: %{y:,.0f}<extra></extra>",
                                  name="Predictions"))
    lim = max(test["crime_count"].max(), test["predicted"].max()) * 1.05
    scatter.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                  line=dict(color=ACCENT, dash="dash"),
                                  name="Perfect prediction"))
    scatter.update_layout(title="<b>Predicted vs Actual</b> (test set)",
                           xaxis_title="Actual crime count", yaxis_title="Predicted crime count",
                           margin=dict(l=0, r=0, t=50, b=0))

    test_copy = test.copy()
    test_copy["abs_error"] = (test_copy["crime_count"] - test_copy["predicted"]).abs()
    bor_err = test_copy.groupby("borough").agg(
        mae=("abs_error", "mean"), avg_actual=("crime_count", "mean")).reset_index()
    bor_err["pct_error"] = (bor_err["mae"] / bor_err["avg_actual"] * 100).round(1)
    bor_err = bor_err.sort_values("pct_error")
    err_chart = go.Figure(go.Bar(
        x=bor_err["pct_error"], y=bor_err["borough"], orientation="h",
        marker_color=[ACCENT if b == "Westminster" else PRIMARY for b in bor_err["borough"]],
        hovertemplate="<b>%{y}</b><br>Error: %{x:.1f}%<extra></extra>"))
    err_chart.update_layout(title="<b>Per-borough prediction error %</b> (test set)",
                             xaxis_title="Mean absolute error (% of average)",
                             height=750, margin=dict(l=0, r=0, t=50, b=0))

    return [
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Test RMSE", className="text-muted"),
                html.H2(f"{rmse:.1f}", style={"color": PRIMARY}),
                html.Small("Root mean squared error", className="text-muted"),
            ]), className="shadow-sm text-center"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Test MAE", className="text-muted"),
                html.H2(f"{mae:.1f}", style={"color": "#2E5FA3"}),
                html.Small("Mean absolute error", className="text-muted"),
            ]), className="shadow-sm text-center"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Test R²", className="text-muted"),
                html.H2(f"{r2:.3f}", style={"color": "#52B788"}),
                html.Small("Variance explained", className="text-muted"),
            ]), className="shadow-sm text-center"), md=4),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(figure=scatter), md=6),
                 dbc.Col(dcc.Graph(figure=err_chart), md=6)]),
        dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            html.B("Note: "), "Westminster (highlighted in red) has higher absolute error ",
            "due to its extreme crime volume, but its percentage error is comparable to other boroughs."
        ], color="info"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

import dash

app = dash.Dash(__name__)
server = app.server

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Starting dashboard server...")
    print("  Open http://127.0.0.1:8050 in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    app.run(debug=False, port=8050)
