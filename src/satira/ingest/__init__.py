"""Data ingestion package: scrapers and source adapters."""

from satira.ingest.base_scraper import BaseScraper, ScrapedItem, ScraperStats
from satira.ingest.image_pipeline import ImageDownloader, ProcessedItem
from satira.ingest.news_scrapers import (
    GDELTScraper,
    NewsScraperRegistry,
    RSSNewsScraper,
)
from satira.ingest.satire_scrapers import (
    BabylonBeeScraper,
    ReductressScraper,
    SatireScraperRegistry,
    TheOnionScraper,
)

__all__ = [
    "BabylonBeeScraper",
    "BaseScraper",
    "GDELTScraper",
    "ImageDownloader",
    "NewsScraperRegistry",
    "ProcessedItem",
    "ReductressScraper",
    "RSSNewsScraper",
    "SatireScraperRegistry",
    "ScrapedItem",
    "ScraperStats",
    "TheOnionScraper",
]
