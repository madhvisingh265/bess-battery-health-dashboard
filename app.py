import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="BESS Dashboard", layout="wide")

st.title("🔋 BESS Battery Health Monitoring Dashboard")

# =========================
# LOAD DATA + MODEL
# =========================

model = joblib.load("model/bess_model.pkl")
features = joblib.load("model/features.pkl")

df = pd.read_csv("outputs/processed_data.csv")

# =========================
# SHOW DATA
# =========================

st.subheader("📊 Processed Dataset")
st.dataframe(df.head())

# =========================
# METRICS
# =========================

latest = df.iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("SoH (%)", f"{latest['soh']:.2f}")
col2.metric("Capacity Fade (%)", f"{latest['capacity_fade']:.2f}")
col3.metric("RUL (cycles)", int(latest["rul_cycles"]))

st.metric("Health Status", latest["health_status"])

# =========================
# GRAPHS
# =========================

st.subheader("📈 SoH vs Cycle")
st.line_chart(df.set_index("cycle_number")["soh"])

st.subheader("📉 Capacity Fade")
st.line_chart(df.set_index("cycle_number")["capacity_fade"])

st.subheader("🔋 RUL Trend")
st.line_chart(df.set_index("cycle_number")["rul_cycles"])

# =========================
# PREDICTION
# =========================

st.subheader("🔮 Predict Battery SoH")

user_input = {}

for feature in features:
    user_input[feature] = st.number_input(feature, value=float(df[feature].mean()))

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted SoH: {prediction:.2f}%")

    if prediction >= 85:
        status = "Healthy"
    elif prediction >= 70:
        status = "Warning"
    elif prediction >= 60:
        status = "Critical"
    else:
        status = "Emergency"

    st.warning(f"Battery Status: {status}")