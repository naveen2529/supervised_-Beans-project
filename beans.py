import streamlit as st
import numpy as np
import joblib

# Load model pipeline and label encoder
pipeline = joblib.load("gradient_boosting_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define the 16 input features (must match training)
feature_names = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1',
    'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]

# UI layout
st.set_page_config(page_title="Bean Classifier", layout="centered")
st.title("🌱 Bean Type Classifier (Gradient Boosting)")
st.markdown("Provide morphological features to predict the bean variety.")

# Collect user inputs
input_values = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0, format="%.4f")
    input_values.append(val)

# Predict button
if st.button("Predict"):
    try:
        input_array = np.array(input_values).reshape(1, -1)
        prediction_encoded = pipeline.predict(input_array)[0]
        predicted_class = label_encoder.inverse_transform([prediction_encoded])[0]
        st.success(f" Predicted Bean Type: **{predicted_class}**")
    except Exception as e:
        st.error(f" Prediction error: {e}")
