"""Named-entity recognition for scraped content.

The bridge between scraper output and the entity graph: spaCy's NER
pipeline runs over each :class:`ProcessedItem`'s text, every surface
form is looked up in the :class:`MentionNormalizer` (Tier 1), resolved
mentions become ``Content -> Entity`` edges, unresolved ones are
queued for Tier 2 batch resolution, and every item gets a
``Content -> Source`` edge so downstream stages can attribute mentions
to publishers.

The extractor accepts an injected ``nlp`` object so tests can run
without a spaCy model on disk; in production the default
``en_core_web_sm`` model is loaded eagerly so missing-model failures
surface at construction rather than mid-pipeline.

Install the spaCy model with ``py -m spacy download en_core_web_sm``
on Windows or ``python -m spacy download en_core_web_sm`` elsewhere.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from satira.graph.entity_resolution import MentionNormalizer
from satira.graph.schema import ContentNode, EdgeType, SourceNode
from satira.graph.store import GraphStore
from satira.ingest.image_pipeline import ProcessedItem


logger = logging.getLogger(__name__)


# Map spaCy NER labels onto the graph's ``entity_type`` vocabulary so a
# pending mention pushed to Tier 2 carries a type that BatchResolver's
# 0.15-weight type term can match without further translation.
_SPACY_TO_TYPE = {
    "PERSON": "person",
    "ORG": "organization",
    "NORP": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "PRODUCT": "product",
    "EVENT": "event",
    "WORK_OF_ART": "product",
}

_DEFAULT_LABELS = frozenset(_SPACY_TO_TYPE)


@dataclass
class ExtractedEntity:
    """A surface-form entity span pulled from a single document."""

    text: str
    label: str
    start: int
    end: int


class EntityExtractor:
    """Run spaCy NER over scraped text and feed results into the graph."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        allowed_labels: Iterable[str] | None = _DEFAULT_LABELS,
        nlp: Any = None,
    ) -> None:
        if nlp is None:
            try:
                import spacy
            except ImportError as exc:
                raise RuntimeError(
                    "spaCy is required for EntityExtractor — install with "
                    "`pip install spacy`"
                ) from exc
            try:
                nlp = spacy.load(model_name)
            except OSError as exc:
                raise RuntimeError(
                    f"spaCy model {model_name!r} not found — run "
                    f"`py -m spacy download {model_name}`"
                ) from exc
        self._nlp = nlp
        self.model_name = model_name
        self.allowed_labels = (
            None if allowed_labels is None else frozenset(allowed_labels)
        )

    def extract(self, text: str) -> list[ExtractedEntity]:
        """Run NER over ``text`` and return matching entity spans."""
        if not text:
            return []
        doc = self._nlp(text)
        return list(self._iter_entities(doc))

    def extract_batch(self, texts: list[str]) -> list[list[ExtractedEntity]]:
        """Batched NER. Preserves input order, even when texts contain blanks.

        spaCy's ``pipe`` is materially faster than calling the model per
        document, so we feed all non-empty texts through a single
        ``pipe`` call and slot the results back into the original
        positions.
        """
        results: list[list[ExtractedEntity]] = [[] for _ in texts]
        nonempty_indices: list[int] = []
        nonempty_texts: list[str] = []
        for i, t in enumerate(texts):
            if t:
                nonempty_indices.append(i)
                nonempty_texts.append(t)
        if not nonempty_texts:
            return results
        for idx, doc in zip(nonempty_indices, self._nlp.pipe(nonempty_texts)):
            results[idx] = list(self._iter_entities(doc))
        return results

    def populate_graph(
        self,
        items: list[ProcessedItem],
        graph_store: GraphStore,
        mention_normalizer: MentionNormalizer,
    ) -> dict:
        """Add content and source nodes for ``items`` and wire up mention edges.

        Resolved mentions become ``Content -> Entity`` (MENTIONS) edges;
        unresolved ones are returned in the ``pending_mentions`` list,
        ready to feed into :class:`BatchResolver`. Every item gets a
        ``Content -> Source`` (POSTED_BY) edge.

        Items whose ``content_id`` already exists in the graph are
        skipped — the same scrape could land in this method twice
        across overlapping ingest windows and that must not double-count
        mentions or raise.
        """
        resolved = 0
        content_added = 0
        pending: list[dict] = []

        # One NER pass over the whole batch is much cheaper than a per-item
        # ``extract`` call: spaCy's pipeline amortises model overhead across
        # documents.
        all_entities = self.extract_batch([it.text for it in items])

        for item, entities in zip(items, all_entities):
            content_id = _content_id(item)
            source_id = _source_id(item)

            # Idempotent source insertion: many items share publishers and
            # we may be invoked across overlapping batches.
            if source_id not in graph_store._nodes:
                graph_store.add_source(SourceNode(
                    id=source_id,
                    domain=item.source_domain,
                    account_id=item.metadata.get("account_id"),
                    credibility_label=item.metadata.get("label", "unknown"),
                ))

            if content_id in graph_store._nodes:
                logger.debug("content %s already present — skipping", content_id)
                continue

            graph_store.add_content(ContentNode(
                id=content_id,
                image_hash=item.perceptual_hash,
                extracted_text=item.text,
                timestamp=item.timestamp,
                source_id=source_id,
            ))
            content_added += 1
            graph_store.add_edge(content_id, source_id, EdgeType.POSTED_BY)

            # Build cooccurrence list as we resolve so unresolved mentions
            # in this same document can use it as a Tier 2 blocking key.
            cooc_entity_ids: list[str] = []
            unresolved: list[ExtractedEntity] = []
            for ent in entities:
                eid, _score = mention_normalizer.normalize(ent.text)
                if eid is not None:
                    if eid not in cooc_entity_ids:
                        cooc_entity_ids.append(eid)
                    graph_store.add_edge(content_id, eid, EdgeType.MENTIONS)
                    resolved += 1
                else:
                    unresolved.append(ent)

            for ent in unresolved:
                pending.append({
                    "text": ent.text,
                    "entity_type": _SPACY_TO_TYPE.get(ent.label),
                    "content_id": content_id,
                    "cooccurring_entity_ids": list(cooc_entity_ids),
                })

        return {
            "entities_resolved": resolved,
            "entities_pending": len(pending),
            "content_added": content_added,
            "pending_mentions": pending,
        }

    def _iter_entities(self, doc: Any) -> Iterable[ExtractedEntity]:
        for ent in doc.ents:
            if self.allowed_labels is not None and ent.label_ not in self.allowed_labels:
                continue
            yield ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            )


def _content_id(item: ProcessedItem) -> str:
    digest = hashlib.sha1(item.source_url.encode("utf-8")).hexdigest()
    return f"content:{digest[:16]}"


def _source_id(item: ProcessedItem) -> str:
    parts = [item.source_domain or "unknown"]
    account_id = item.metadata.get("account_id") if item.metadata else None
    if account_id:
        parts.append(str(account_id))
    return "source:" + ":".join(parts)
