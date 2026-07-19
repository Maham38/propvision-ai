from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.predict import load_model, predict_house_price


MODEL_PATH = "models/best_house_price_model.joblib"


st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1f2937 100%);
        color: #f9fafb;
    }
    .main-card {
        background: rgba(17, 24, 39, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("House Price Prediction System")
st.caption("Train on Zameen.com Islamabad listings and estimate a house price from key features.")

model_file = Path(MODEL_PATH)
if model_file.exists():
    model_name, _ = load_model(MODEL_PATH)
    st.info(f"Loaded trained model: {model_name}")
else:
    st.warning("No trained model found yet. Run the training script first.")

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (marla)", min_value=1.0, value=10.0, step=0.5)
        bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, value=3, step=1)
        built_in_year = st.number_input("Built-in Year", min_value=1900, max_value=2035, value=2015, step=1)
        property_type = st.selectbox("Property Type", ["House", "Flat", "Upper Portion", "Lower Portion", "Penthouse", "Farm House"], index=0)

    with col2:
        location = st.text_input("Location", value="DHA Defence")
        city = st.selectbox("City", ["Islamabad"], index=0)
        parking_spaces = st.number_input("Parking Spaces", min_value=0, value=1, step=1)
        servant_quarters = st.number_input("Servant Quarters", min_value=0, value=0, step=1)
        store_rooms = st.number_input("Store Rooms", min_value=0, value=0, step=1)
        kitchens = st.number_input("Kitchens", min_value=0, value=1, step=1)
        drawing_rooms = st.number_input("Drawing Rooms", min_value=0, value=1, step=1)

    predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    if not model_file.exists():
        st.error("Train the model first so the app can load a saved pipeline.")
    else:
        estimated_price = predict_house_price(
            area=area,
            bedrooms=int(bedrooms),
            bathrooms=int(bathrooms),
            location=location,
            city=city,
            property_type=property_type,
            built_in_year=int(built_in_year),
            parking_spaces=int(parking_spaces),
            servant_quarters=int(servant_quarters),
            store_rooms=int(store_rooms),
            kitchens=int(kitchens),
            drawing_rooms=int(drawing_rooms),
            model_path=MODEL_PATH,
        )
        st.success(f"Estimated House Price: PKR {estimated_price:,.0f}")
