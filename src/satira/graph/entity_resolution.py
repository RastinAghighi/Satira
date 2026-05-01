from dataclasses import dataclass

from satira.graph.store import GraphStore


_HIGH_CONFIDENCE_THRESHOLD = 0.92
_MAX_FUZZY_CANDIDATES = 50
_PREFIX_LEN = 3


_GRAPH_WEIGHT_BY_TYPE = {
    "exact_alias": 1.0,
    "high_confidence": 0.85,
    "provisional": 0.5,
    "unresolved": 0.0,
}


@dataclass
class EntityResolutionResult:
    entity_id: str | None
    confidence: float
    resolution_type: str  # exact_alias, high_confidence, provisional, unresolved

    @property
    def graph_weight(self) -> float:
        """Weight multiplier for the graph embedding attention pooling."""
        return _GRAPH_WEIGHT_BY_TYPE.get(self.resolution_type, 0.0)


class MentionNormalizer:
    """Tier 1: Hot path entity resolution — sub-millisecond per mention.

    Handles ~80-85% of incoming mentions via in-memory alias lookup.

    Uses:
    1. Exact alias lookup (dict) after lowercasing/stripping
    2. Prefix bucket lookup with edit distance for fuzzy matches
    3. Unresolved mentions go to pending queue for Tier 2
    """

    def __init__(self) -> None:
        self.alias_map: dict[str, str] = {}
        self.canonical_names: dict[str, str] = {}
        self._prefix_buckets: dict[str, list[tuple[str, str]]] = {}

    # --- population -----------------------------------------------------
    def load_from_graph(self, graph_store: GraphStore) -> None:
        """Populate alias_map and canonical_names from all entities and aliases."""
        for node_id, kind in graph_store._node_kind.items():
            if kind != "entity":
                continue
            entity = graph_store._nodes[node_id]
            self.canonical_names[entity.id] = entity.canonical_name
            self.register_alias(entity.canonical_name, entity.id)
            for alias in entity.aliases:
                self.register_alias(alias, entity.id)

    def register_alias(self, alias: str, entity_id: str) -> None:
        """Add new alias discovered during Tier 2 resolution."""
        cleaned = _clean(alias)
        if not cleaned:
            return
        self.alias_map[cleaned] = entity_id
        bucket_key = cleaned[:_PREFIX_LEN]
        bucket = self._prefix_buckets.setdefault(bucket_key, [])
        for existing_alias, existing_id in bucket:
            if existing_alias == cleaned and existing_id == entity_id:
                return
        bucket.append((cleaned, entity_id))

    # --- lookup ---------------------------------------------------------
    def normalize(self, raw_mention: str) -> tuple[str | None, float]:
        cleaned = _clean(raw_mention)
        if not cleaned:
            return (None, 0.0)

        exact = self.alias_map.get(cleaned)
        if exact is not None:
            return (exact, 1.0)

        bucket_key = cleaned[:_PREFIX_LEN]
        bucket = self._prefix_buckets.get(bucket_key, [])
        if not bucket:
            return (None, 0.0)

        best_id: str | None = None
        best_score = 0.0
        for alias, entity_id in bucket[:_MAX_FUZZY_CANDIDATES]:
            score = _similarity(cleaned, alias)
            if score > best_score:
                best_score = score
                best_id = entity_id
                if score >= 1.0:
                    break

        if best_score >= _HIGH_CONFIDENCE_THRESHOLD:
            return (best_id, best_score)
        return (None, 0.0)

    # --- introspection --------------------------------------------------
    def stats(self) -> dict:
        return {
            "total_aliases": len(self.alias_map),
            "total_entities": len(self.canonical_names),
            "bucket_count": len(self._prefix_buckets),
        }


def _clean(text: str) -> str:
    return text.strip().lower()


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    distance = _edit_distance(a, b)
    return 1.0 - distance / longest


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[-1]
