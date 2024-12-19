import streamlit as st
import pickle
import gzip
import joblib as jb
import numpy as np

# Load the trained model
with gzip.open('random_forest_model.pkl.gz', 'rb') as f:
    rf_model = jb.load(f)

# App title
st.title("Student Performance Prediction")

# Input fields for the features
hours_studied = st.number_input("Hours Studied", min_value=0.0, step=0.1)
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, step=0.1)
extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, step=0.1)
sample_papers = st.number_input("Sample Papers Practiced", min_value=0, step=1)

# Convert categorical input to numerical
extracurricular_encoded = 1 if extracurricular == "Yes" else 0

# Predict button
if st.button("Predict"):
    # Prepare the input data
    features = np.array([[hours_studied, previous_score, extracurricular_encoded, sleep_hours, sample_papers]])
    
    # Make prediction
    prediction = rf_model.predict(features)
    
    # Display the result
    st.success(f"Predicted Score: {prediction[0]:.2f}")
