"""Data ingestion package: scrapers and source adapters."""

from satira.ingest.archive_scrapers import (
    AIOptOutVerdict,
    ArchiveScraper,
    ArchiveScraperRegistry,
    ArchiveScraperState,
    FlatSitemapArchiveScraper,
    PaginatedArchiveScraper,
    SitemapIndexArchiveScraper,
    build_scraper_from_config,
    detect_ai_optout,
)
from satira.ingest.base_scraper import BaseScraper, ScrapedItem, ScraperStats
from satira.ingest.entity_extraction import EntityExtractor, ExtractedEntity
from satira.ingest.huggingface_loader import (
    KNOWN_SATIRE_DATASETS,
    HFDatasetLoader,
    HFDatasetSpec,
)
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
from satira.ingest.source_credibility import (
    KNOWN_NEWS,
    KNOWN_SATIRE,
    SourceClassification,
    SourceCredibilityClassifier,
)

__all__ = [
    "AIOptOutVerdict",
    "ArchiveScraper",
    "ArchiveScraperRegistry",
    "ArchiveScraperState",
    "BabylonBeeScraper",
    "BaseScraper",
    "EntityExtractor",
    "ExtractedEntity",
    "FlatSitemapArchiveScraper",
    "GDELTScraper",
    "HFDatasetLoader",
    "HFDatasetSpec",
    "ImageDownloader",
    "KNOWN_NEWS",
    "KNOWN_SATIRE",
    "KNOWN_SATIRE_DATASETS",
    "NewsScraperRegistry",
    "PaginatedArchiveScraper",
    "ProcessedItem",
    "ReductressScraper",
    "RSSNewsScraper",
    "SatireScraperRegistry",
    "ScrapedItem",
    "ScraperStats",
    "SitemapIndexArchiveScraper",
    "SourceClassification",
    "SourceCredibilityClassifier",
    "TheOnionScraper",
    "build_scraper_from_config",
    "detect_ai_optout",
]
