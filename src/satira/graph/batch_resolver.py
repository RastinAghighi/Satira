import math
import random
from typing import Iterable

from satira.graph.schema import EntityNode
from satira.graph.store import GraphStore


class BatchResolver:
    """Tier 2: Warm path — batch clustering of unresolved mentions every 5-10 minutes.

    Uses three blocking strategies in parallel to narrow candidate sets:
    1. FirstTokenBlock: mentions sharing first token
    2. EmbeddingLSHBlock: semantically similar mentions via locality-sensitive hashing
    3. CooccurrenceBlock: entities mentioned in the same document

    Within each block, computes weighted similarity:
    - Embedding: 0.4
    - String:    0.25
    - Co-occurrence: 0.2
    - Type:      0.15

    Three-threshold decision:
    - Above MERGE_THRESHOLD (0.85): auto-merge
    - Between MERGE and REVIEW (0.65-0.85): flag for human review
    - Below REVIEW (0.65): create new entity
    """

    EMBED_W = 0.4
    STRING_W = 0.25
    COOC_W = 0.2
    TYPE_W = 0.15

    def __init__(
        self,
        graph_store: GraphStore,
        merge_threshold: float = 0.85,
        review_threshold: float = 0.65,
        entity_embeddings: dict[str, list[float]] | None = None,
        lsh_bits: int = 8,
        lsh_seed: int = 0,
    ) -> None:
        if review_threshold > merge_threshold:
            raise ValueError("review_threshold must be <= merge_threshold")
        self.graph_store = graph_store
        self.merge_threshold = merge_threshold
        self.review_threshold = review_threshold
        self.entity_embeddings: dict[str, list[float]] = dict(entity_embeddings or {})
        self._lsh = _LSH(bits=lsh_bits, seed=lsh_seed)
        self.last_comparison_count = 0
        self.last_naive_comparison_count = 0

    # --- public API -----------------------------------------------------
    def resolve_batch(self, pending_mentions: list[dict]) -> list[dict]:
        blocks = self._build_blocks(pending_mentions)

        candidates_per_mention: dict[int, set[str]] = {
            i: set() for i in range(len(pending_mentions))
        }
        for record_list in blocks.values():
            mention_indices = [r["_idx"] for r in record_list if r["_kind"] == "mention"]
            entity_ids = {r["id"] for r in record_list if r["_kind"] == "entity"}
            for idx in mention_indices:
                candidates_per_mention[idx].update(entity_ids)

        all_entity_ids = [
            nid for nid, kind in self.graph_store._node_kind.items() if kind == "entity"
        ]
        self.last_naive_comparison_count = len(pending_mentions) * len(all_entity_ids)
        self.last_comparison_count = sum(len(s) for s in candidates_per_mention.values())

        decisions: list[dict] = []
        for idx, mention in enumerate(pending_mentions):
            best_score = 0.0
            best_id: str | None = None
            for entity_id in candidates_per_mention[idx]:
                entity = self.graph_store.get_entity(entity_id)
                if entity is None:
                    continue
                score = self._compute_similarity(mention, entity)
                if score > best_score:
                    best_score = score
                    best_id = entity_id

            if best_id is not None and best_score >= self.merge_threshold:
                action, target = "merge", best_id
            elif best_id is not None and best_score >= self.review_threshold:
                action, target = "review", best_id
            else:
                action, target = "create", None

            decisions.append({
                "mention": mention.get("text", ""),
                "action": action,
                "target_entity": target,
                "score": best_score,
            })
        return decisions

    def _build_blocks(self, mentions: list[dict]) -> dict[str, list[dict]]:
        blocks: dict[str, list[dict]] = {}

        # 1. FirstToken — both sides
        for idx, mention in enumerate(mentions):
            tok = _first_token(mention.get("text", ""))
            if tok:
                blocks.setdefault(f"ft:{tok}", []).append(
                    {"_kind": "mention", "_idx": idx}
                )
        for entity in self._iter_entities():
            seen_tokens: set[str] = set()
            for name in (entity.canonical_name, *entity.aliases):
                tok = _first_token(name)
                if not tok or tok in seen_tokens:
                    continue
                seen_tokens.add(tok)
                blocks.setdefault(f"ft:{tok}", []).append(
                    {"_kind": "entity", "id": entity.id}
                )

        # 2. EmbeddingLSH
        for idx, mention in enumerate(mentions):
            emb = mention.get("embedding")
            if emb is None:
                continue
            sig = self._lsh.signature(emb)
            blocks.setdefault(f"lsh:{sig}", []).append(
                {"_kind": "mention", "_idx": idx}
            )
        for entity_id, emb in self.entity_embeddings.items():
            sig = self._lsh.signature(emb)
            blocks.setdefault(f"lsh:{sig}", []).append(
                {"_kind": "entity", "id": entity_id}
            )

        # 3. Cooccurrence — mentions reaching out to the entities they sit beside
        for idx, mention in enumerate(mentions):
            for cooc_id in mention.get("cooccurring_entity_ids", []) or []:
                key = f"cooc:{cooc_id}"
                blocks.setdefault(key, []).append({"_kind": "mention", "_idx": idx})
                blocks.setdefault(key, []).append({"_kind": "entity", "id": cooc_id})

        return blocks

    def _compute_similarity(self, mention: dict, entity: EntityNode) -> float:
        # String similarity: best across canonical + aliases
        m_text = (mention.get("text") or "").strip().lower()
        names = [entity.canonical_name.lower(), *(a.lower() for a in entity.aliases)]
        if m_text and names:
            string_sim = max(_string_similarity(m_text, n) for n in names)
        else:
            string_sim = 0.0

        # Embedding similarity
        m_emb = mention.get("embedding")
        e_emb = self.entity_embeddings.get(entity.id)
        if m_emb is not None and e_emb is not None:
            embed_sim = _cosine(m_emb, e_emb)
        else:
            embed_sim = 0.0

        # Co-occurrence: binary signal
        cooc_ids = mention.get("cooccurring_entity_ids") or []
        cooc_sim = 1.0 if entity.id in cooc_ids else 0.0

        # Type
        m_type = mention.get("entity_type")
        type_sim = 1.0 if m_type and m_type == entity.entity_type else 0.0

        return (
            self.EMBED_W * embed_sim
            + self.STRING_W * string_sim
            + self.COOC_W * cooc_sim
            + self.TYPE_W * type_sim
        )

    # --- helpers --------------------------------------------------------
    def _iter_entities(self) -> Iterable[EntityNode]:
        for nid, kind in self.graph_store._node_kind.items():
            if kind == "entity":
                yield self.graph_store._nodes[nid]


class MergeTracker:
    """Tracks which graph nodes were affected by merges and need GNN re-embedding."""

    def __init__(self) -> None:
        self._merges: list[dict] = []
        self._affected: set[str] = set()

    def record_merge(
        self,
        source_id: str,
        target_id: str,
        affected_edges: Iterable,
    ) -> None:
        edges = list(affected_edges)
        self._merges.append({
            "source_id": source_id,
            "target_id": target_id,
            "affected_edges": edges,
        })
        self._affected.add(source_id)
        self._affected.add(target_id)
        for edge in edges:
            if isinstance(edge, str):
                self._affected.add(edge)
            elif isinstance(edge, (tuple, list)):
                for part in edge:
                    if isinstance(part, str):
                        self._affected.add(part)

    def get_priority_recompute_set(self) -> set[str]:
        return set(self._affected)

    def clear(self) -> None:
        self._merges.clear()
        self._affected.clear()


# --- internals ----------------------------------------------------------
class _LSH:
    """Hyperplane LSH: sign of dot product against `bits` random unit vectors."""

    def __init__(self, bits: int = 8, seed: int = 0) -> None:
        self.bits = bits
        self.seed = seed
        self._planes: list[list[float]] | None = None

    def signature(self, vec: list[float]) -> str:
        if self._planes is None or len(self._planes[0]) != len(vec):
            rng = random.Random(self.seed)
            self._planes = [
                [rng.gauss(0.0, 1.0) for _ in range(len(vec))] for _ in range(self.bits)
            ]
        out = []
        for plane in self._planes:
            dot = sum(p * v for p, v in zip(plane, vec))
            out.append("1" if dot >= 0.0 else "0")
        return "".join(out)


def _first_token(text: str) -> str:
    if not text:
        return ""
    return text.strip().lower().split()[0] if text.strip() else ""


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _string_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - _edit_distance(a, b) / longest


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
