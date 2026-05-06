"""Data ingestion package: scrapers and source adapters."""

from satira.ingest.base_scraper import BaseScraper, ScrapedItem, ScraperStats
from satira.ingest.satire_scrapers import (
    BabylonBeeScraper,
    ReductressScraper,
    SatireScraperRegistry,
    TheOnionScraper,
)

__all__ = [
    "BabylonBeeScraper",
    "BaseScraper",
    "ReductressScraper",
    "SatireScraperRegistry",
    "ScrapedItem",
    "ScraperStats",
    "TheOnionScraper",
]
