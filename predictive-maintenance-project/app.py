# app.py - UPGRADED WITH REAL MODEL
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
from datetime import datetime
import joblib

# Set page config
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🏭 Predictive Maintenance Dashboard")
st.markdown("Real-time monitoring of IoT-connected machines for early failure detection.")

# Load model, scaler, and features
# ================
# 🚨 ROBUST MODEL LOADER — Auto-generates data & trains model if missing
# ================
@st.cache_resource
def load_model():
    # Check if model and scaler exist
    if not (os.path.exists('models/xgb_failure_model.pkl') and
            os.path.exists('models/scaler.pkl') and
            os.path.exists('models/feature_names.pkl')):

        st.warning("⚠️ Model not found. Training now... (this may take 30-60 seconds)")
        
        # Auto-generate dataset if missing
        if not os.path.exists('data/iot_sensor_data_8500.csv'):
            st.info("🔄 Generating dataset...")
            import generate_iot_data
            generate_iot_data.main()  # We'll add a main() function below

        # Auto-train model
        st.info("🧠 Training XGBoost model with SMOTE...")
        import save_model
        save_model.main()  # We'll add a main() function below

    # Now load the model (should exist)
    model = joblib.load('models/xgb_failure_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/feature_names.pkl')
    return model, scaler, features

model, scaler, features = load_model()

# Load data
df = pd.read_csv('data/iot_sensor_data_8500.csv')

# Feature Engineering (MUST match training)
df['sensor_01_rolling_avg'] = df['sensor_01'].rolling(window=10, min_periods=1).mean()
df['sensor_03_spike'] = (df['sensor_03'] > df['sensor_03'].rolling(window=10, min_periods=1).mean() * 2).astype(int)
df['sensor_05_std_last_5'] = df['sensor_05'].rolling(window=5, min_periods=1).std().fillna(0)

# Sidebar
st.sidebar.header("⚙️ Controls")
machine_id = st.sidebar.selectbox("Select Machine", df['machine_id'].unique())

# Filter data
machine_data = df[df['machine_id'] == machine_id].copy()

# Make predictions (for last 100 records only to avoid lag)
if len(machine_data) > 100:
    machine_data = machine_data.tail(100)

# Prepare features for prediction
X_pred = machine_data[features].values
X_pred_scaled = scaler.transform(X_pred)
failure_probs = model.predict_proba(X_pred_scaled)[:, 1]  # Probability of failure

# Add predictions to dataframe
machine_data['failure_prob'] = failure_probs

# Plot 1: Sensor Trends + Failure Probability
st.subheader(f"📊 Sensor Trends & Failure Probability - {machine_id}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['sensor_01'], name='Vibration', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['sensor_02'], name='Temperature', line=dict(color='red')))
fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['sensor_03'], name='Pressure', line=dict(color='green')))
fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['failure_prob']*10, name='Failure Risk (x10)', line=dict(color='orange', dash='dot')))

fig.update_layout(title="Sensor Readings + Failure Probability", xaxis_title="Time", yaxis_title="Value")
st.plotly_chart(fig, use_container_width=True)

# Plot 2: Current Risk Score
st.subheader("🚨 Current Risk Assessment")
current_risk = failure_probs[-1] if len(failure_probs) > 0 else 0
risk_level = "🟢 Low" if current_risk < 0.3 else "🟡 Medium" if current_risk < 0.7 else "🔴 High"

col1, col2, col3 = st.columns(3)
col1.metric("Failure Probability", f"{current_risk:.2%}")
col2.metric("Risk Level", risk_level)
col3.metric("Last Prediction", datetime.now().strftime("%H:%M:%S"))

# Plot 3: Alert History
st.subheader("🔔 Recent High-Risk Events")
high_risk = machine_data[machine_data['failure_prob'] > 0.5]
if len(high_risk) > 0:
    fig_alerts = px.scatter(high_risk, x='timestamp', y='failure_prob', color='failure_prob',
                           title="High-Risk Events (Probability > 50%)",
                           color_continuous_scale='Reds')
    st.plotly_chart(fig_alerts, use_container_width=True)
else:
    st.info("No high-risk events detected in recent data.")

# Add footer
st.markdown("---")
st.caption("💡 Powered by XGBoost + Streamlit | Real-time predictive maintenance for industrial IoT")
