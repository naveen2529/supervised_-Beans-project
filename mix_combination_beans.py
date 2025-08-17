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

# Streamlit UI
st.set_page_config(page_title="Bean Classifier", layout="centered")
st.title("🌱 Bean Type Classifier (Gradient Boosting)")
st.markdown("Provide morphological features to predict the bean variety.")

# Collect user inputs
input_values = []

# 1. Area, Perimeter → sliders (since usually bounded ranges)
area = st.slider("Area", min_value=0.0, max_value=20000.0, value=5000.0, step=100.0)
perimeter = st.slider("Perimeter", min_value=0.0, max_value=1000.0, value=200.0, step=10.0)

# 2. Major & Minor Axis → number input
major_axis = st.number_input("Major Axis Length", value=400.0, format="%.4f")
minor_axis = st.number_input("Minor Axis Length", value=150.0, format="%.4f")

# 3. Aspect Ratio & Eccentricity → slider
aspect_ratio = st.slider("Aspect Ratio", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
eccentricity = st.slider("Eccentricity", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# 4. Convex Area & EquivDiameter → number inputs
convex_area = st.number_input("Convex Area", value=5100.0, format="%.4f")
equiv_diameter = st.number_input("Equivalent Diameter", value=80.0, format="%.4f")

# 5. Extent & Solidity → sliders
extent = st.slider("Extent", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
solidity = st.slider("Solidity", min_value=0.0, max_value=1.0, value=0.90, step=0.01)

# 6. Roundness & Compactness → radio (3 levels: Low, Medium, High)
roundness_level = st.radio("Roundness", ["Low", "Medium", "High"], index=1)
compactness_level = st.radio("Compactness", ["Low", "Medium", "High"], index=1)

roundness = {"Low": 0.3, "Medium": 0.6, "High": 0.9}[roundness_level]
compactness = {"Low": 0.3, "Medium": 0.6, "High": 0.9}[compactness_level]

# 7. Shape Factors (all as number inputs)
shape1 = st.number_input("Shape Factor 1", value=0.5, format="%.4f")
shape2 = st.number_input("Shape Factor 2", value=0.5, format="%.4f")
shape3 = st.number_input("Shape Factor 3", value=0.5, format="%.4f")
shape4 = st.number_input("Shape Factor 4", value=0.5, format="%.4f")

# Collect all inputs in correct order
input_values = [
    area, perimeter, major_axis, minor_axis,
    aspect_ratio, eccentricity, convex_area, equiv_diameter,
    extent, solidity, roundness, compactness, shape1, shape2, shape3, shape4
]

# Prediction
if st.button("🔍 Predict Bean Type"):
    try:
        input_array = np.array(input_values).reshape(1, -1)
        prediction_encoded = pipeline.predict(input_array)[0]
        predicted_class = label_encoder.inverse_transform([prediction_encoded])[0]
        st.success(f"✅ Predicted Bean Type: **{predicted_class}**")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
