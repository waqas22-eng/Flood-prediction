import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🌊 Flood Risk Prediction System")
st.markdown("### AI-based Flood Risk Analysis with Map & Dashboard")

# SIDEBAR INPUTS
st.sidebar.header("Enter Environmental Data")

rainfall = st.sidebar.slider("Rainfall (mm)", 0, 300, 100)
river_level = st.sidebar.slider("River Level", 0, 10, 5)
elevation = st.sidebar.slider("Elevation (m)", 0, 1000, 300)
distance = st.sidebar.slider("Distance from River (km)", 0, 20, 5)
history = st.sidebar.selectbox("Flood History", [0, 1])

# LOCATION INPUT (FOR MAP)
latitude = st.sidebar.number_input("Latitude", value=34.0)
longitude = st.sidebar.number_input("Longitude", value=72.0)

# PREDICTION BUTTON
if st.sidebar.button("Predict"):

    # Prepare data
    data = np.array([[rainfall, river_level, elevation, distance, history]])

    # Model prediction
    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

    # RESULT
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ HIGH FLOOD RISK")
    else:
        st.success("✅ LOW FLOOD RISK")

    # Probability
    st.write(f"📈 Flood Probability: {probability*100:.2f}%")

    # Progress bar
    st.progress(int(probability * 100))

    # ALERT SYSTEM
    if probability > 0.7:
        st.warning("🚨 ALERT: Immediate action required!")
    elif probability > 0.5:
        st.info("⚠️ Medium risk. Stay alert.")
    else:
        st.success("✅ Situation is safe.")

    # ---------------- GRAPH ----------------
    st.subheader("📊 Input Data Visualization")

    labels = ['Rainfall', 'River Level', 'Elevation', 'Distance']
    values = [rainfall, river_level, elevation, distance]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)

    # Pie chart
    st.subheader("📈 Risk Distribution")

    fig2, ax2 = plt.subplots()
    ax2.pie([probability, 1 - probability],
            labels=["Flood Risk", "Safe"],
            autopct='%1.1f%%')
    st.pyplot(fig2)

    # ---------------- MAP ----------------
    st.subheader("🗺️ Flood Risk Location")

    map_data = pd.DataFrame({
        'lat': [latitude],
        'lon': [longitude]
    })

    st.map(map_data)