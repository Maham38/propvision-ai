from __future__ import annotations

import joblib

from .preprocess import build_prediction_frame


def load_model(model_path: str = "models/best_house_price_model.joblib"):
    artifact = joblib.load(model_path)
    return artifact["model_name"], artifact["pipeline"]


def predict_house_price(
    area: float,
    bedrooms: int,
    bathrooms: int,
    location: str,
    city: str = "Islamabad",
    property_type: str = "House",
    built_in_year: int = 2015,
    parking_spaces: int = 1,
    servant_quarters: int = 0,
    store_rooms: int = 0,
    kitchens: int = 1,
    drawing_rooms: int = 1,
    model_path: str = "models/best_house_price_model.joblib",
) -> float:
    _, pipeline = load_model(model_path)
    features = build_prediction_frame(
        area=area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        location=location,
        city=city,
        property_type=property_type,
        built_in_year=built_in_year,
        parking_spaces=parking_spaces,
        servant_quarters=servant_quarters,
        store_rooms=store_rooms,
        kitchens=kitchens,
        drawing_rooms=drawing_rooms,
    )
    prediction = pipeline.predict(features)[0]
    return float(prediction)
