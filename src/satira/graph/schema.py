from dataclasses import dataclass
from datetime import datetime
from enum import Enum


@dataclass
class EntityNode:
    id: str
    canonical_name: str
    entity_type: str  # person, organization, product, location
    aliases: list[str]
    created_at: datetime


@dataclass
class SourceNode:
    id: str
    domain: str
    account_id: str | None
    credibility_label: str  # satire, news, mixed, unknown


@dataclass
class EventNode:
    id: str
    topic_cluster: str
    date_range: tuple[datetime, datetime]


@dataclass
class TemplateNode:
    id: str
    perceptual_hash: str
    layout_features: list[float]


@dataclass
class ContentNode:
    id: str
    image_hash: str
    extracted_text: str
    timestamp: datetime
    source_id: str | None


class EdgeType(Enum):
    MENTIONS = "mentions"           # Content -> Entity
    POSTED_BY = "posted_by"         # Content -> Source
    REFERENCES = "references"       # Content -> Event
    USES = "uses"                   # Content -> Template
    INVOLVED_IN = "involved_in"     # Entity -> Event
