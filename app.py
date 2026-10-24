import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "wattwise_random_forest.joblib")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "hourly_energy_data.csv")

st.set_page_config(
    page_title="WattWise AI",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    data = data.sort_values("datetime").reset_index(drop=True)
    return data

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    st.error("The model or processed dataset is missing.")
    st.stop()

bundle = load_model()
model = bundle["model"]
features = bundle["features"]
target = bundle["target"]
df = load_data()

st.sidebar.title("⚡ WattWise AI")
st.sidebar.caption("Smart Energy Intelligence Platform")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Energy Analytics", "Forecast Lab", "Bill Estimator"]
)

st.title("⚡ WattWise AI")
st.caption("Understand energy usage, forecast consumption, and estimate electricity costs.")

latest = df.iloc[-1]
average_power = df[target].mean()
peak_power = df[target].max()

if page == "Overview":
    st.subheader("Energy Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Active Power", f"{average_power:.2f} kW")
    c2.metric("Peak Hourly Average", f"{peak_power:.2f} kW")
    c3.metric("Hourly Records", f"{len(df):,}")

    chart_df = df.tail(24 * 7)
    fig = px.line(
        chart_df,
        x="datetime",
        y=target,
        title="Hourly Average Power — Last Available Week",
        labels={"datetime": "Time", target: "Active power (kW)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This project uses historical household electricity readings. "
        "Its predictions are estimates, not guaranteed future consumption."
    )

elif page == "Energy Analytics":
    st.subheader("Explore Energy Usage")

    max_days = max(1, min(90, len(df) // 24))
    days = st.slider("Days to display", 1, max_days, min(7, max_days))
    view = df.tail(days * 24)

    fig = px.line(
        view, x="datetime", y=target,
        title=f"Active Power Over the Last {days} Days",
        labels={"datetime": "Time", target: "Active power (kW)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    if "Sub_metering_1" in view.columns:
        sub_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        sub = view[["datetime"] + sub_cols].melt(
            id_vars="datetime", var_name="Circuit", value_name="Reading"
        )
        fig2 = px.line(
            sub, x="datetime", y="Reading", color="Circuit",
            title="Sub-metering Readings"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(view.tail(20), use_container_width=True)

elif page == "Forecast Lab":
    st.subheader("One-Hour-Ahead Forecast")

    if len(df) < 169:
        st.warning("There is not enough historical data for the lag features.")
    else:
        history = df.tail(168).copy()
        values = history[target].to_numpy(dtype=float)

        # Use the latest observed hour to predict the next hour.
        next_time = pd.to_datetime(history["datetime"].iloc[-1]) + pd.Timedelta(hours=1)

        row = {
            "hour": next_time.hour,
            "day_of_week": next_time.dayofweek,
            "day_of_month": next_time.day,
            "month": next_time.month,
            "is_weekend": int(next_time.dayofweek >= 5),
            "power_lag_1": values[-1],
            "power_lag_2": values[-2],
            "power_lag_24": values[-24],
            "power_lag_168": values[-168],
            "rolling_mean_24": values[-24:].mean(),
            "rolling_mean_168": values.mean()
        }

        X_next = pd.DataFrame([row])[features]
        prediction = float(model.predict(X_next)[0])

        c1, c2 = st.columns(2)
        c1.metric("Predicted Active Power", f"{prediction:.3f} kW")
        c2.metric("Forecast Time", next_time.strftime("%d %b %Y, %H:%M"))

        st.caption(
            "This is a one-hour-ahead prediction using the latest available "
            "historical readings. It is not a multi-day forecast."
        )

        hist = history.tail(48)
        fig = px.line(
            hist, x="datetime", y=target,
            title="Recent Observed Power",
            labels={"datetime": "Time", target: "Active power (kW)"}
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Bill Estimator":
    st.subheader("Electricity Bill Estimator")

    tariff = st.number_input(
        "Electricity rate (currency units per kWh)",
        min_value=0.0, value=7.0, step=0.5
    )
    period = st.selectbox("Billing period", ["Last 7 days", "Last 30 days"])

    hours = 24 * (7 if period == "Last 7 days" else 30)
    usage = df.tail(hours)
    estimated_kwh = float(usage[target].sum())
    estimated_cost = estimated_kwh * tariff

    c1, c2 = st.columns(2)
    c1.metric("Estimated Energy", f"{estimated_kwh:,.2f} kWh")
    c2.metric("Estimated Energy Cost", f"{estimated_cost:,.2f}")

    st.caption(
        "Illustrative estimate based on the dataset's hourly average active "
        "power and your entered rate. It excludes fixed charges, taxes, "
        "time-of-use rates, and other billing adjustments."
    )

st.sidebar.divider()
st.sidebar.caption("WattWise AI • Machine Learning Project")
