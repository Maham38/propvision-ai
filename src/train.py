from __future__ import annotations

import argparse
from pathlib import Path
from math import sqrt

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .preprocess import FEATURE_COLUMNS, clean_dataset


try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = ["city", "location", "property_type"]


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, object]:
    models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror",
        )

    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
        )

    return models


def evaluate_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, Pipeline]]:
    features = frame[FEATURE_COLUMNS].copy()
    target = frame["price"].copy()

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    preprocessor = make_preprocessor()
    results: list[dict[str, float | str]] = []
    best_name = ""
    best_model: Pipeline | None = None
    best_rmse = float("inf")

    for name, estimator in build_models().items():
        model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, predictions)

        results.append({"model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No model could be trained successfully.")

    metrics_frame = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    return metrics_frame, (best_name, best_model)


def train_and_save(data_path: str, model_dir: str) -> pd.DataFrame:
    raw_frame = pd.read_csv(data_path)
    frame = clean_dataset(raw_frame)
    if frame.empty:
        raise ValueError("The dataset is empty after preprocessing.")

    metrics_frame, (best_name, best_model) = evaluate_models(frame)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_house_price_model.joblib"
    joblib.dump({"model_name": best_name, "pipeline": best_model}, model_path)

    metrics_path = output_dir / "model_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    return metrics_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare house price prediction models")
    parser.add_argument("--data", default="data/islamabad_listings.csv", help="Path to the cleaned CSV dataset")
    parser.add_argument("--model-dir", default="models", help="Directory to store the trained model and metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_frame = train_and_save(args.data, args.model_dir)
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
