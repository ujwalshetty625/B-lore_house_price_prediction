import streamlit as st
import numpy as np
import pandas as pd
import json
import joblib

# Load model
model = joblib.load('house_price_model.pkl')

# Load columns
with open("columns.json", "r") as f:
    data_columns = json.load(f)

# Get locations and area types from columns
locations = [col.replace("location_", "") for col in data_columns if col.startswith("location_")]
area_types = [col.replace("area_type_", "") for col in data_columns if col.startswith("area_type_")]

# Streamlit UI
st.set_page_config(page_title="🏠 Bangalore House Price Predictor", layout="centered")

st.title("🏠 Bangalore House Price Predictor")
st.markdown("##### Enter property details below to estimate the house price (in Lakhs).")

# Input Fields
sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, value=1000)
bath = st.selectbox("Bathrooms", [1, 2, 3, 4, 5])
bhk = st.selectbox("BHK (Bedrooms)", [1, 2, 3, 4, 5])
balcony = st.selectbox("Balcony", [0, 1, 2, 3])
location = st.selectbox("Location", sorted(locations))
area_type = st.selectbox("Area Type", sorted(area_types))

# Predict Button
if st.button("Predict Price"):
    input_data = np.zeros(len(data_columns))

    # Set core numeric features
    if "total_sqft" in data_columns:
        input_data[data_columns.index("total_sqft")] = sqft
    if "bath" in data_columns:
        input_data[data_columns.index("bath")] = bath
    if "bhk" in data_columns:
        input_data[data_columns.index("bhk")] = bhk
    if "balcony" in data_columns:
        input_data[data_columns.index("balcony")] = balcony

    # Set one-hot encoded location
    location_key = f"location_{location}"
    if location_key in data_columns:
        input_data[data_columns.index(location_key)] = 1

    # Set one-hot encoded area type
    area_key = f"area_type_{area_type}"
    if area_key in data_columns:
        input_data[data_columns.index(area_key)] = 1

    # Predict
    predicted_price = model.predict([input_data])[0]

    st.success(f"🏡 Estimated Price: ₹ {predicted_price:.2f} Lakhs")
st.markdown("---")
st.info("📌 **Note:** This prediction is based on a demo model trained on limited data and may not reflect real-world property prices accurately.")

