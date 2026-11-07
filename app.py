"""
Melbourne Housing Price Predictor
A Streamlit app powered by a scikit-learn pipeline trained on Melbourne housing sales.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "housing_model.joblib"
METADATA_PATH = ROOT / "model" / "model_metadata.json"

st.set_page_config(
    page_title="Melbourne Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


model, meta = load_artifacts()
mae = meta["expected_error"]
metrics = meta["metrics"][meta["best_model"]]["test"]
type_labels = meta["type_labels"]
code_by_label = {label: code for code, label in type_labels.items()}

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1180px;}
        .hero {
            background: linear-gradient(120deg, #0f172a 0%, #134e4a 60%, #0f766e 100%);
            border-radius: 18px;
            padding: 34px 38px 30px 38px;
            color: #f8fafc;
            margin-bottom: 26px;
        }
        .hero h1 {font-size: 2.15rem; margin: 0 0 8px 0; font-weight: 700; letter-spacing: -0.02em;}
        .hero p {margin: 0; color: #cbd5e1; font-size: 1.02rem; max-width: 760px;}
        .hero .tag {
            display: inline-block; margin-bottom: 14px; padding: 4px 12px;
            border: 1px solid rgba(255,255,255,0.28); border-radius: 999px;
            font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: #99f6e4;
        }
        .stat-card {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
            padding: 16px 18px; height: 100%;
        }
        .stat-label {color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;}
        .stat-value {color: #0f172a; font-size: 1.55rem; font-weight: 700; margin-top: 4px;}
        .stat-note {color: #94a3b8; font-size: 0.78rem; margin-top: 2px;}
        .result-card {
            background: linear-gradient(150deg, #ecfdf5 0%, #f0fdfa 55%, #ffffff 100%);
            border: 1px solid #99f6e4; border-radius: 18px;
            padding: 26px 28px; margin-bottom: 16px;
        }
        .result-label {color: #0f766e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;}
        .result-price {color: #0f172a; font-size: 2.7rem; font-weight: 800; letter-spacing: -0.03em; margin: 6px 0 2px 0;}
        .result-range {color: #475569; font-size: 0.95rem;}
        .result-note {color: #94a3b8; font-size: 0.78rem; margin-top: 12px;}
        .snapshot {
            background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px;
            padding: 18px 20px; color: #334155; font-size: 0.92rem; line-height: 1.75;
        }
        .snapshot b {color: #0f172a;}
        .footer {color: #94a3b8; font-size: 0.8rem; margin-top: 30px; text-align: center;}
        div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <div class="tag">Melbourne · Property Valuation</div>
        <h1>Melbourne Housing Price Predictor</h1>
        <p>
            Estimate the market value of a Melbourne property from its suburb, size and
            location. Built with a scikit-learn {meta['best_model'].lower()} pipeline trained
            on {meta['dataset_rows']:,} historic sales.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

stat_cols = st.columns(3)
stats = [
    ("Model accuracy (R²)", f"{metrics['r2']:.2f}", "on unseen test data"),
    ("Typical error (MAE)", f"${mae / 1000:,.0f}k", "mean absolute error"),
    ("Training records", f"{meta['dataset_rows']:,}", "cleaned Melbourne sales"),
]
for column, (label, value, note) in zip(stat_cols, stats):
    with column:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

predict_tab, performance_tab, about_tab = st.tabs(
    ["Predict a price", "Model performance", "About this project"]
)

with predict_tab:
    input_col, result_col = st.columns([1.25, 0.9], gap="large")

    with input_col:
        st.subheader("Property details")
        st.caption("Change any field and the estimate updates instantly.")

        suburb = st.selectbox(
            "Suburb",
            options=meta["suburbs"],
            index=meta["suburbs"].index("Melbourne")
            if "Melbourne" in meta["suburbs"]
            else 0,
        )
        defaults = meta["suburb_defaults"][suburb]

        type_col, rooms_col = st.columns(2)
        with type_col:
            type_options = list(code_by_label)
            type_label = st.selectbox(
                "Property type",
                options=type_options,
                index=type_options.index(type_labels[defaults["type"]]),
                key=f"type_{suburb}",
            )
        with rooms_col:
            rooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)

        distance = st.slider(
            "Distance from Melbourne CBD (km)",
            min_value=0.0,
            max_value=60.0,
            value=float(defaults["distance"]),
            step=0.1,
            key=f"distance_{suburb}",
        )

        with st.expander("Advanced location details"):
            postcode = st.selectbox(
                "Postcode",
                options=meta["postcodes"],
                index=meta["postcodes"].index(defaults["postcode"]),
                key=f"postcode_{suburb}",
            )
            region = st.selectbox(
                "Region",
                options=meta["regions"],
                index=meta["regions"].index(defaults["region"]),
                key=f"region_{suburb}",
            )
            property_count = st.number_input(
                "Properties in suburb",
                min_value=100,
                max_value=25000,
                value=int(defaults["propertycount"]),
                step=100,
                key=f"propertycount_{suburb}",
                help="Total number of properties recorded for this suburb.",
            )

        st.caption(
            "Suburb, postcode and region are pre-filled with the most common values "
            "for the selected suburb. You can override them in the advanced section."
        )

    type_code = code_by_label[type_label]
    property_row = pd.DataFrame(
        [
            {
                "Suburb": suburb,
                "Type": type_code,
                "Regionname": region,
                "Postcode": postcode,
                "Rooms": rooms,
                "Distance": distance,
                "Propertycount": property_count,
            }
        ]
    )

    prediction = float(model.predict(property_row)[0])
    prediction = max(prediction, 50_000)
    rounded = round(prediction, -3)
    low = max(round(prediction - mae, -3), 0)
    high = round(prediction + mae, -3)

    suburb_median = defaults["median_price"]
    difference = prediction - suburb_median
    comparison = (
        f"{difference / suburb_median:+.0%} vs the suburb median"
        if suburb_median
        else "no suburb median available"
    )

    with result_col:
        st.subheader("Estimated value")
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated market price</div>
                <div class="result-price">${rounded:,.0f}</div>
                <div class="result-range">Likely range: ${low:,.0f} – ${high:,.0f}</div>
                <div class="result-note">
                    Range reflects the model's average test error of ±${mae:,.0f}.
                </div>
            </div>
            <div class="snapshot">
                <b>{suburb}</b> · {type_label} · {rooms} bed<br>
                {distance:.1f} km from the CBD · Postcode {postcode}<br>
                Suburb median sale price: <b>${suburb_median:,.0f}</b><br>
                This estimate is <b>{comparison}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Estimates are indicative only. They are based on historic sale patterns "
            "and do not account for renovations, views, or current market conditions.",
            icon="ℹ️",
        )

with performance_tab:
    st.subheader("How the model performs")
    st.caption(
        "Three algorithms were trained on an 80/20 train-test split of the cleaned "
        f"dataset ({meta['split']['train_rows']:,} training rows, "
        f"{meta['split']['test_rows']:,} test rows). The best model is deployed."
    )

    rows = []
    for name, result in meta["metrics"].items():
        test = result["test"]
        rows.append(
            {
                "Model": name,
                "Test RMSE": f"${test['rmse']:,.0f}",
                "Test MAE": f"${test['mae']:,.0f}",
                "R²": f"{test['r2']:.3f}",
                "MAPE": f"{test['mape_pct']:.1f}%",
                "Fit time (s)": result["fit_seconds"],
            }
        )
    performance_df = pd.DataFrame(rows)
    st.dataframe(performance_df, hide_index=True)

    st.markdown(
        f"""
        **Deployed model:** {meta['best_model']}  
        **Trained:** {meta['trained_at'][:10]}  
        **Features used:** suburb, property type, region, postcode, bedrooms,
        distance from the CBD, and number of properties in the suburb.
        """
    )
    st.caption(
        "R² measures the share of price variation explained by the model. "
        "MAE is the average absolute dollar error on the test set."
    )

with about_tab:
    st.subheader("About this project")
    st.markdown(
        f"""
        This app is an end-to-end machine learning deployment:

        1. **Data** — {meta['dataset_rows']:,} Melbourne property sales with price,
           location and property attributes.
        2. **Features** — suburb, type, region, postcode, bedrooms, distance from the
           CBD, and suburb property count.
        3. **Pipeline** — missing-value imputation, one-hot encoding and scaling are
           bundled with the estimator in a single scikit-learn pipeline, so training
           and serving behave identically.
        4. **Model selection** — Linear Regression, Random Forest and Gradient Boosting
           are compared on a held-out test set; the best R² wins.
        5. **Serving** — Streamlit Community Cloud runs `app.py` and loads the saved
           `model/housing_model.joblib` artifact.

        **Limitations:** the dataset is a historic snapshot and excludes features such as
        land size, building area, year built and condition. Predictions should be treated
        as a guide, not a professional valuation.
        """
    )
    st.markdown(
        f"""
        **Price range in training data:** ${meta['price']['min']:,} – ${meta['price']['max']:,}  
        **Median price:** ${meta['price']['median']:,}
        """
    )

st.markdown(
    '<div class="footer">Built with scikit-learn and Streamlit · Data: Melbourne housing sales</div>',
    unsafe_allow_html=True,
)
