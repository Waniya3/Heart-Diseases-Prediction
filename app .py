import streamlit as st
import pickle
import numpy as np
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "heart_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

st.title("❤️ Heart Disease Prediction")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0,1])
chest_pain = st.number_input("Chest Pain Type")
bp = st.number_input("Blood Pressure")
cholesterol = st.number_input("Cholesterol")
fbs = st.number_input("Fasting Blood Sugar")
ecg = st.number_input("ECG Results")
max_hr = st.number_input("Maximum Heart Rate")
exercise = st.number_input("Exercise Angina")
oldpeak = st.number_input("Oldpeak")
slope = st.number_input("Slope")
vessels = st.number_input("Number of Vessels")
thal = st.number_input("Thal")

if st.button("Predict"):

    data = np.array([[age, sex, chest_pain, bp,
                      cholesterol, fbs, ecg,
                      max_hr, exercise, oldpeak,
                      slope, vessels, thal]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")
