from __future__ import annotations

import re

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "price",
    "area",
    "bedrooms",
    "bathrooms",
    "built_in_year",
    "parking_spaces",
    "servant_quarters",
    "store_rooms",
    "kitchens",
    "drawing_rooms",
]

CATEGORICAL_COLUMNS = ["city", "location", "property_type"]

FEATURE_COLUMNS = [
    "area",
    "city",
    "bedrooms",
    "bathrooms",
    "location",
    "property_type",
    "built_in_year",
    "parking_spaces",
    "servant_quarters",
    "store_rooms",
    "kitchens",
    "drawing_rooms",
]


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_")
        for column in frame.columns
    }
    return frame.rename(columns=renamed)


def parse_numeric(value: object) -> float | np.nan:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).lower().replace(",", "").strip()
    if not text or text in {"nan", "none", "unknown", "-"}:
        return np.nan

    multipliers = {
        "crore": 10_000_000,
        "lac": 100_000,
        "lakh": 100_000,
        "thousand": 1_000,
        "k": 1_000,
    }
    for token, multiplier in multipliers.items():
        if token in text:
            match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", text)
            return float(match.group(1)) * multiplier if match else np.nan

    match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", text)
    return float(match.group(1)) if match else np.nan


def parse_area_to_marla(value: object) -> float | np.nan:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).lower().replace(",", "").strip()
    if not text or text in {"nan", "none", "unknown", "-"}:
        return np.nan

    match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", text)
    if not match:
        return np.nan

    quantity = float(match.group(1))
    if "kanal" in text:
        return quantity * 20
    if "sq ft" in text or "sqft" in text:
        return quantity / 272.25
    if "sq yd" in text or "sqyd" in text:
        return quantity / 30.25
    return quantity


def coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()

    if "price" in cleaned.columns:
        cleaned["price"] = cleaned["price"].map(parse_numeric)

    if "area" in cleaned.columns:
        cleaned["area"] = cleaned["area"].map(parse_area_to_marla)

    for column in [c for c in NUMERIC_COLUMNS if c in cleaned.columns and c not in {"price", "area"}]:
        cleaned[column] = cleaned[column].map(parse_numeric)

    return cleaned


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = standardize_columns(frame)
    cleaned = coerce_numeric_columns(cleaned)

    for column in CATEGORICAL_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = np.nan

    for column in FEATURE_COLUMNS + ["price"]:
        if column not in cleaned.columns:
            cleaned[column] = np.nan

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in [c for c in NUMERIC_COLUMNS if c in cleaned.columns]:
        if column == "price":
            continue
        if cleaned[column].notna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip()

    cleaned["city"] = cleaned["city"].replace({"": "Unknown"})
    cleaned["location"] = cleaned["location"].replace({"": "Unknown"})
    cleaned["property_type"] = cleaned["property_type"].replace({"": "Unknown"})

    if "built_in_year" in cleaned.columns:
        year_median = cleaned["built_in_year"].dropna().median()
        cleaned["built_in_year"] = cleaned["built_in_year"].fillna(year_median if pd.notna(year_median) else 0)
        cleaned["built_in_year"] = cleaned["built_in_year"].round().astype(int)

    for column in ["bedrooms", "bathrooms", "parking_spaces", "servant_quarters", "store_rooms", "kitchens", "drawing_rooms"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(0).round().astype(int)

    if "price" in cleaned.columns:
        cleaned = cleaned[cleaned["price"].notna()]

    cleaned = cleaned[cleaned["area"].notna()]
    return cleaned.reset_index(drop=True)


def ensure_feature_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_dataset(raw_frame)
    for column in FEATURE_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = 0 if column in {"area", "bedrooms", "bathrooms", "built_in_year", "parking_spaces", "servant_quarters", "store_rooms", "kitchens", "drawing_rooms"} else "Unknown"
    return cleaned[FEATURE_COLUMNS + ["price"]].copy()


def build_prediction_frame(
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
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "area": float(area),
                "city": city,
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "location": location,
                "property_type": property_type,
                "built_in_year": int(built_in_year),
                "parking_spaces": int(parking_spaces),
                "servant_quarters": int(servant_quarters),
                "store_rooms": int(store_rooms),
                "kitchens": int(kitchens),
                "drawing_rooms": int(drawing_rooms),
            }
        ]
    )
