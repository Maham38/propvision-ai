from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class ListingRecord:
    price: str | None = None
    area: str | None = None
    city: str | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    location: str | None = None
    property_type: str | None = None
    built_in_year: int | None = None
    parking_spaces: int | None = None
    servant_quarters: int | None = None
    store_rooms: int | None = None
    kitchens: int | None = None
    drawing_rooms: int | None = None
    source_url: str | None = None


def fetch_html(url: str, session: requests.Session, timeout: int = 20) -> str:
    response = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def build_page_url(start_url: str, page_number: int, page_url_template: str | None = None) -> str:
    if page_url_template:
        return page_url_template.format(page=page_number, start_url=start_url)

    if page_number <= 1:
        return start_url

    parsed = urlparse(start_url)
    query = parse_qs(parsed.query)
    query["page"] = [str(page_number)]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def extract_json_ld(soup: BeautifulSoup) -> dict:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            parsed = json.loads(script.string or script.get_text(strip=True))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
    return {}


def extract_text_by_label(soup: BeautifulSoup, label_candidates: Iterable[str]) -> str | None:
    lowered_candidates = [candidate.lower() for candidate in label_candidates]
    for element in soup.find_all(string=True):
        text = element.strip()
        lower = text.lower()
        if any(lower == candidate or lower.startswith(candidate + " ") for candidate in lowered_candidates):
            sibling = element.parent.find_next_sibling()
            if sibling:
                value = sibling.get_text(" ", strip=True)
                if value:
                    return value
    return None


def extract_number_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"([0-9]+)", text.replace(",", ""))
    return int(match.group(1)) if match else None


def extract_listing_links(search_html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(search_html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if "/property/" not in href.lower() and "/properties/" not in href.lower():
            continue
        absolute = urljoin(base_url, href)
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def parse_listing_detail(detail_html: str, source_url: str) -> ListingRecord:
    soup = BeautifulSoup(detail_html, "html.parser")
    data = extract_json_ld(soup)

    price = None
    if isinstance(data.get("offers"), dict):
        price = data["offers"].get("price")
    if price is None:
        price = extract_text_by_label(soup, ["price"])

    title = data.get("name") or extract_text_by_label(soup, ["title"])
    location = extract_text_by_label(soup, ["location", "address"])
    city = "Islamabad"
    if location and "," in location:
        city = location.split(",")[-1].strip() or city

    property_type = None
    if title:
        property_type = title.split("for")[-1].strip() if "for" in title.lower() else title
    if not property_type:
        property_type = extract_text_by_label(soup, ["property type", "type"])

    area = extract_text_by_label(soup, ["area", "size"])
    bedrooms = extract_number_from_text(extract_text_by_label(soup, ["bedrooms", "beds"]))
    bathrooms = extract_number_from_text(extract_text_by_label(soup, ["bathrooms", "baths"]))
    built_in_year = extract_number_from_text(extract_text_by_label(soup, ["year", "built in year"]))
    parking_spaces = extract_number_from_text(extract_text_by_label(soup, ["parking", "parking spaces"]))
    servant_quarters = extract_number_from_text(extract_text_by_label(soup, ["servant quarters"]))
    store_rooms = extract_number_from_text(extract_text_by_label(soup, ["store rooms", "store room"]))
    kitchens = extract_number_from_text(extract_text_by_label(soup, ["kitchens", "kitchen"]))
    drawing_rooms = extract_number_from_text(extract_text_by_label(soup, ["drawing rooms", "drawing room"]))

    return ListingRecord(
        price=str(price) if price is not None else None,
        area=area,
        city=city,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        location=location,
        property_type=property_type,
        built_in_year=built_in_year,
        parking_spaces=parking_spaces,
        servant_quarters=servant_quarters,
        store_rooms=store_rooms,
        kitchens=kitchens,
        drawing_rooms=drawing_rooms,
        source_url=source_url,
    )


def scrape_islamabad_listings(
    start_url: str,
    max_listings: int = 350,
    delay_seconds: float = 1.0,
    page_url_template: str | None = None,
) -> pd.DataFrame:
    session = requests.Session()
    collected: list[ListingRecord] = []
    seen_links: set[str] = set()

    page_number = 1
    while len(collected) < max_listings:
        page_url = build_page_url(start_url, page_number, page_url_template)
        LOGGER.info("Fetching page %s", page_url)
        html = fetch_html(page_url, session)
        listing_links = extract_listing_links(html, page_url)

        if not listing_links:
            LOGGER.info("No more listing links found on page %s", page_number)
            break

        new_links = [link for link in listing_links if link not in seen_links]
        if not new_links:
            LOGGER.info("Pagination appears exhausted at page %s", page_number)
            break

        for link in new_links:
            if len(collected) >= max_listings:
                break
            try:
                detail_html = fetch_html(link, session)
                record = parse_listing_detail(detail_html, link)
                collected.append(record)
                seen_links.add(link)
                LOGGER.info("Collected %s / %s", len(collected), max_listings)
                time.sleep(delay_seconds)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Skipping %s due to error: %s", link, exc)

        page_number += 1
        time.sleep(delay_seconds)

    frame = pd.DataFrame([asdict(record) for record in collected])
    return frame


def save_csv(frame: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Islamabad house listings from Zameen.com")
    parser.add_argument("--start-url", required=True, help="Islamabad search result URL on Zameen")
    parser.add_argument("--output", default="data/islamabad_listings.csv", help="CSV output path")
    parser.add_argument("--max-listings", type=int, default=350, help="Maximum number of listings to collect")
    parser.add_argument("--delay-seconds", type=float, default=1.0, help="Delay between requests")
    parser.add_argument(
        "--page-url-template",
        default=None,
        help="Optional pagination template such as https://example.com/results?page={page}",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    frame = scrape_islamabad_listings(args.start_url, args.max_listings, args.delay_seconds, args.page_url_template)
    if frame.empty:
        raise RuntimeError("No listings were collected. Check the start URL or the site markup.")
    save_csv(frame, args.output)
    LOGGER.info("Saved %s rows to %s", len(frame), args.output)


if __name__ == "__main__":
    main()
