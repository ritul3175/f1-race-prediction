import base64
import os
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from data_loader import (
    get_practice_laps,
    get_event_schedule,
    get_driver_info,
    get_race_results,
)
from feature_engineering import preprocess_laps, aggregate_driver_features
from models import (
    train_qualifying_model,
    predict_qualifying,
    train_race_model,
    predict_race,
    compute_race_accuracy,
)

st.set_page_config(page_title="F1 Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Background image and custom styling ---
def get_base64_of_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_IMG_PATH = os.path.join(os.path.dirname(__file__), "images.png")
_bg_b64 = get_base64_of_image(_IMG_PATH) if os.path.isfile(_IMG_PATH) else None
_bg_css = f'url("data:image/png;base64,{_bg_b64}") center center no-repeat' if _bg_b64 else "none"

st.markdown(
    f"""
    <style>
    /* Full-page background with your image */
    .stApp {{
        background: linear-gradient(135deg, rgba(0,0,0,0.88) 0%, rgba(20,20,20,0.9) 50%, rgba(0,0,0,0.88) 100%),
                    {_bg_css};
        background-size: cover;
        background-attachment: fixed;
    }}
    /* Main content area - glass-style card */
    .main .block-container {{
        padding: 2rem 3rem 3rem;
        max-width: 1100px;
    }}
    /* Headers */
    h1 {{
        color: #e10600 !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        text-shadow: 0 0 20px rgba(225, 6, 0, 0.3);
        margin-bottom: 0.5rem !important;
    }}
    h2, h3 {{
        color: #fff !important;
        font-weight: 600 !important;
    }}
    /* Labels and text */
    label, p, .stMarkdown {{
        color: #e0e0e0 !important;
    }}
    /* Select boxes and inputs - dark with red accent */
    .stSelectbox > div > div {{
        background-color: rgba(30,30,30,0.95) !important;
        border: 1px solid rgba(225, 6, 0, 0.4) !important;
        border-radius: 8px !important;
    }}
    /* Buttons - F1 red */
    .stButton > button {{
        background: linear-gradient(180deg, #e10600 0%, #b80500 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(225, 6, 0, 0.5) !important;
    }}
    /* Tabs - racing style */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background: rgba(25,25,25,0.9);
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid rgba(225, 6, 0, 0.25);
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #aaa !important;
        border-radius: 8px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(225, 6, 0, 0.2) !important;
        color: #e10600 !important;
    }}
    /* DataFrames - dark cards */
    .stDataFrame {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(225, 6, 0, 0.2);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: #e10600 !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: #bbb !important;
    }}
    /* Radio options */
    .stRadio > div {{
        background: rgba(30,30,30,0.6);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(225, 6, 0, 0.15);
    }}
    /* Alerts / info boxes */
    .stAlert {{
        border-radius: 8px;
        border-left: 4px solid #e10600;
    }}
    /* Spinner text */
    .stSpinner label {{
        color: #e10600 !important;
    }}
    /* Compact season/venue row - smaller select boxes */
    [data-testid="column"] .stSelectbox {{
        max-width: 280px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏎️ Formula 1 Qualifying & Race Prediction")
st.caption("Predict lap times and race order from practice data • XGBoost-powered")

# Season and venue selection side by side in compact boxes
@st.cache_data(show_spinner="Loading race calendar...")
def load_venues(y):
    return get_event_schedule(y)

col_season, col_venue = st.columns([1, 2])  # venue wider for long GP names
with col_season:
    # Allow seasons from 2018 up to and including 2026
    year = st.selectbox("Select Season", list(range(2018, 2027)), key="year")
with col_venue:
    try:
        venues = load_venues(year)
    except ValueError as e:
        st.error(str(e))
        st.info("Try again in a moment, or choose a different year (e.g. 2024).")
        st.stop()
    if not venues:
        st.warning("No events found for this season.")
        st.stop()
    venue = st.selectbox("Select Grand Prix", venues, key="venue")

# Main tabs: Predictions | Accuracy
tab_predictions, tab_accuracy = st.tabs(["Predictions", "Model Accuracy"])

# --- Tab 1: Predictions ---
with tab_predictions:
    prediction_type = st.radio(
        "Prediction Type",
        ["Qualifying Lap Time", "Race Outcome"],
        key="pred_type",
    )

    if st.button("Run Prediction", key="run_pred"):
        try:
            with st.spinner("Loading practice data (this may take a minute the first time)..."):
                laps = get_practice_laps(year, venue)
                driver_map = get_driver_info(year, venue)
        except ValueError as e:
            st.error(str(e))
            st.info("Choose an event whose practice sessions have taken place and are available in FastF1.")
            st.stop()

        df = preprocess_laps(laps)
        df = aggregate_driver_features(df)

        if driver_map:
            df["Driver"] = df["Driver"].map(lambda x: driver_map.get(x, x))

        if prediction_type == "Qualifying Lap Time":
            model, _, _, _ = train_qualifying_model(df)
            result = predict_qualifying(model, df)
            st.caption("Predictions from XGBoost trained on practice pace features.")

            # Show how prediction differs from best practice lap
            if "LapTime_min" in result.columns:
                result["BestPracticeLapTime"] = result["LapTime_min"]
                result["PredictedDelta"] = result["PredictedLapTime"] - result["BestPracticeLapTime"]
                display_cols = ["Driver", "BestPracticeLapTime", "PredictedLapTime", "PredictedDelta"]
            else:
                display_cols = ["Driver", "PredictedLapTime"]

            st.dataframe(result[display_cols], use_container_width=True)
        else:
            model = train_race_model(df)
            result = predict_race(model, df)
            st.caption("Predicted race order from XGBoost (practice pace). Does not account for strategy, incidents, or DNFs.")
            st.dataframe(result[["Driver", "PredictedPosition"]], use_container_width=True)

# --- Tab 2: Model Accuracy ---
with tab_accuracy:
    accuracy_type = st.radio(
        "Select accuracy to view",
        ["Qualifying", "Race Outcome"],
        key="acc_type",
    )

    if st.button("View Accuracy", key="run_acc"):
        try:
            with st.spinner("Loading data..."):
                laps = get_practice_laps(year, venue)
                driver_map = get_driver_info(year, venue)
        except ValueError as e:
            st.error(str(e))
            st.info("Choose an event whose practice sessions have taken place and are available in FastF1.")
            st.stop()

        df = preprocess_laps(laps)
        df = aggregate_driver_features(df)
        if driver_map:
            df["Driver"] = df["Driver"].map(lambda x: driver_map.get(x, x))

        if accuracy_type == "Qualifying":
            model, mae, y_actual, y_pred = train_qualifying_model(df)
            r2 = r2_score(y_actual, y_pred)

            st.subheader("Qualifying Accuracy")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE (Mean Absolute Error)", f"{mae:.3f} seconds")
            with col2:
                st.metric("R² Score", f"{r2:.4f}")
            st.caption("Metrics from cross-validated predictions (out-of-sample).")

            fig, ax = plt.subplots()
            ax.scatter(y_actual, y_pred, alpha=0.7)
            min_val = min(y_actual.min(), y_pred.min())
            max_val = max(y_actual.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
            ax.set_title("XGBoost Model: Actual vs. Predicted Lap Times")
            ax.set_xlabel("Actual Lap Time (s)")
            ax.set_ylabel("Predicted Lap Time (s)")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:  # Race Outcome
            race_results = get_race_results(year, venue)
            if race_results.empty:
                st.warning("No race results available for this event (race may not have occurred yet).")
            else:
                model = train_race_model(df)
                predicted = predict_race(model, df)
                mae, spearman, merged = compute_race_accuracy(
                    predicted, race_results, driver_map
                )

                st.subheader("Race Outcome Accuracy")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MAE (Mean Absolute Error of Positions)", f"{mae:.2f}")
                with col2:
                    st.metric("Spearman Rank Correlation", f"{spearman:.4f}")
                st.caption("Compares predicted race order with actual finishing positions.")

                if len(merged) >= 2:
                    fig, ax = plt.subplots()
                    ax.scatter(merged["Position"], merged["PredictedPosition"], alpha=0.7)
                    max_p = max(merged["Position"].max(), merged["PredictedPosition"].max())
                    ax.plot([1, max_p], [1, max_p], "r--", label="Perfect prediction")
                    ax.set_title("Race Outcome: Actual vs. Predicted Position")
                    ax.set_xlabel("Actual Finishing Position")
                    ax.set_ylabel("Predicted Position")
                    ax.grid(True)
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                st.subheader("Predicted vs Actual Comparison")
                display_cols = ["Driver", "PredictedPosition", "Position"]
                if "Driver" in merged.columns:
                    merged_display = merged[display_cols].copy()
                    merged_display["Position Error"] = (merged_display["PredictedPosition"] - merged_display["Position"]).abs()
                    st.dataframe(merged_display, use_container_width=True)
