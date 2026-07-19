from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


LOCALITIES = [
    "DHA Defence",
    "Bahria Town",
    "G-13",
    "G-11",
    "G-10",
    "F-10",
    "F-11",
    "E-11",
    "I-8",
    "I-9",
    "B-17",
    "PWD Housing Scheme",
    "Soan Garden",
    "Koral Town",
    "CDA Sector",
]

PROPERTY_TYPES = ["House", "Flat", "Upper Portion", "Lower Portion", "Penthouse", "Farm House"]
YEARS = list(range(1995, 2025))


def area_to_marla_value(area_text: str) -> float:
    text = area_text.lower()
    value = float(text.split()[0])
    if "kanal" in text:
        return value * 20
    if "sq yd" in text:
        return value / 30.25
    if "sq ft" in text:
        return value / 272.25
    return value


def format_area(rng: random.Random) -> str:
    choice = rng.choices(["marla", "kanal", "sq yd", "sq ft"], weights=[58, 18, 14, 10], k=1)[0]
    if choice == "marla":
        return f"{rng.randint(3, 40)} Marla"
    if choice == "kanal":
        return f"{rng.randint(1, 8)} Kanal"
    if choice == "sq yd":
        return f"{rng.randint(100, 1200)} Sq. Yd."
    return f"{rng.randint(900, 8000)} Sq. Ft."


def format_price(price: int) -> str:
    if price >= 10_000_000:
        return f"PKR {price / 10_000_000:.2f} Crore"
    if price >= 100_000:
        return f"PKR {price / 100_000:.2f} Lac"
    return f"PKR {price:,}"


def generate_row(rng: random.Random) -> dict[str, object]:
    city = "Islamabad"
    location = rng.choice(LOCALITIES)
    property_type = rng.choice(PROPERTY_TYPES)
    built_in_year = rng.choice(YEARS)

    area_text = format_area(rng)
    area_marla = area_to_marla_value(area_text)

    bedrooms = max(0, int(round(area_marla / 5 + rng.uniform(-1.5, 2.5))))
    bathrooms = max(1, bedrooms + rng.randint(-1, 2))
    parking_spaces = max(0, min(4, int(round(bedrooms / 2 + rng.uniform(-0.5, 1.5)))))
    servant_quarters = 1 if property_type in {"House", "Farm House"} and area_marla >= 10 and rng.random() < 0.35 else 0
    store_rooms = 1 if area_marla >= 8 and rng.random() < 0.55 else 0
    kitchens = 1 if property_type in {"Flat", "Penthouse"} else max(1, min(3, int(round(bedrooms / 4 + 1))))
    drawing_rooms = 1 if property_type in {"Flat", "Penthouse"} else max(0, min(3, int(round(bedrooms / 3 + rng.uniform(0, 1)))))

    base_price = area_marla * rng.uniform(1_300_000, 2_800_000)
    locality_multiplier = {
        "DHA Defence": 1.65,
        "Bahria Town": 1.35,
        "F-10": 1.8,
        "F-11": 1.9,
        "E-11": 1.45,
        "G-11": 1.25,
        "G-10": 1.22,
        "G-13": 1.18,
        "I-8": 1.28,
        "I-9": 1.15,
        "B-17": 1.0,
        "PWD Housing Scheme": 1.08,
        "Soan Garden": 0.95,
        "Koral Town": 0.9,
        "CDA Sector": 1.3,
    }[location]
    type_multiplier = {
        "House": 1.22,
        "Flat": 0.88,
        "Upper Portion": 0.93,
        "Lower Portion": 0.92,
        "Penthouse": 1.55,
        "Farm House": 1.7,
    }[property_type]

    feature_bonus = (
        bedrooms * 850_000
        + bathrooms * 450_000
        + parking_spaces * 280_000
        + servant_quarters * 650_000
        + store_rooms * 220_000
        + kitchens * 180_000
        + drawing_rooms * 200_000
    )
    age_penalty = max(0, 2026 - built_in_year) * 60_000
    price = int(max(5_500_000, (base_price * locality_multiplier * type_multiplier) + feature_bonus - age_penalty))

    return {
        "price": format_price(price),
        "area": area_text,
        "city": city,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": location,
        "property_type": property_type,
        "built_in_year": built_in_year,
        "parking_spaces": parking_spaces,
        "servant_quarters": servant_quarters,
        "store_rooms": store_rooms,
        "kitchens": kitchens,
        "drawing_rooms": drawing_rooms,
        "source_url": "synthetic://islamabad-local-dataset",
    }


def generate_dataset(rows: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    data = [generate_row(rng) for _ in range(rows)]
    return pd.DataFrame(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local Islamabad house price dataset")
    parser.add_argument("--rows", type=int, default=350, help="Number of rows to generate")
    parser.add_argument("--output", default="data/islamabad_listings.csv", help="CSV output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = generate_dataset(args.rows, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved {len(frame)} rows to {output_path}")


if __name__ == "__main__":
    main()
