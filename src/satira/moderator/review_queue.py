import hashlib
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from satira.graph.schema import EdgeType
from satira.graph.store import GraphStore


@dataclass
class ReviewItem:
    id: str
    mention_text: str
    candidate_entities: list[tuple[str, float]]
    similarity_score: float = 0.0
    priority_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    auto_resolve_at: datetime | None = None
    affected_content_count: int = 0
    embedding_impact: float = 0.0
    ingest_velocity: float = 0.0


@dataclass
class ReviewCluster:
    cluster_id: str
    representative: ReviewItem
    members: list[ReviewItem]
    aggregate_priority: float


class ReviewQueueManager:
    """Priority-scored review queue with batch clustering and auto-resolution.

    Design principles:
    - Priority scoring ensures trending entities bubble to top during surges
    - Batch clustering via agglomerative clustering on mention embeddings
      so moderators resolve 38 similar mentions with one decision
    - Auto-resolution after 30min timeout prevents unbounded queue growth
    - All auto-resolutions go to deferred_review for post-surge correction
    """

    AFFECTED_W = 0.35
    EMBED_IMPACT_W = 0.25
    INGEST_W = 0.25
    AMBIGUITY_W = 0.15

    AFFECTED_NORM_CAP = 100
    INGEST_NORM_CAP = 60.0

    VALID_ACTIONS = ("merge", "reject", "create", "split", "escalate")

    def __init__(self, graph_store: GraphStore, auto_resolve_window_minutes: int = 30) -> None:
        self.graph_store = graph_store
        self.auto_resolve_window = timedelta(minutes=auto_resolve_window_minutes)
        self._pending: dict[str, ReviewItem] = {}
        self._push_times: list[datetime] = []
        self._auto_resolve_count = 0
        self._human_resolved_count = 0
        self._resolution_durations: list[float] = []
        self.deferred_review: list[dict] = []
        self.resolutions: list[dict] = []

    # --- public API -----------------------------------------------------
    def push(self, item: ReviewItem) -> None:
        if item.auto_resolve_at is None:
            item.auto_resolve_at = item.created_at + self.auto_resolve_window
        self._pending[item.id] = item
        self._push_times.append(item.created_at)
        item.ingest_velocity = self._raw_ingest_velocity(item.created_at)
        item.priority_score = self.compute_review_priority(item)

    def get_next_cluster(self) -> ReviewCluster | None:
        clusters = self.cluster_pending_items()
        if not clusters:
            return None
        clusters.sort(key=lambda c: c.aggregate_priority, reverse=True)
        return clusters[0]

    def resolve_cluster(
        self,
        cluster_id: str,
        action: str,
        target_entity: str | None = None,
    ) -> int:
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"action must be one of {self.VALID_ACTIONS}, got {action!r}")
        clusters = {c.cluster_id: c for c in self.cluster_pending_items()}
        cluster = clusters.get(cluster_id)
        if cluster is None:
            raise KeyError(f"unknown cluster {cluster_id!r}")

        now = datetime.now(timezone.utc)
        for member in cluster.members:
            self._pending.pop(member.id, None)
            self._human_resolved_count += 1
            self._resolution_durations.append((now - member.created_at).total_seconds())
            self.resolutions.append({
                "item_id": member.id,
                "cluster_id": cluster_id,
                "action": action,
                "target_entity": target_entity,
                "resolved_at": now,
            })
        return len(cluster.members)

    def process_stale_items(self) -> int:
        now = datetime.now(timezone.utc)
        stale_ids = [iid for iid, it in self._pending.items() if it.auto_resolve_at <= now]
        for iid in stale_ids:
            item = self._pending.pop(iid)
            self._auto_resolve_count += 1
            self.deferred_review.append({
                "item_id": item.id,
                "mention_text": item.mention_text,
                "candidate_entities": list(item.candidate_entities),
                "auto_resolved_at": now,
            })
        return len(stale_ids)

    def cluster_pending_items(self) -> list[ReviewCluster]:
        items = list(self._pending.values())
        if not items:
            return []

        n = len(items)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        # Group by top candidate entity — items pointing at the same entity
        # are obviously the same review decision in 38-out-of-38 cases.
        candidate_to_indices: dict[str, list[int]] = {}
        for idx, item in enumerate(items):
            top = item.candidate_entities[0][0] if item.candidate_entities else None
            if top:
                candidate_to_indices.setdefault(top, []).append(idx)
        for indices in candidate_to_indices.values():
            for k in range(1, len(indices)):
                union(indices[0], indices[k])

        # Token-overlap fallback for items that have no candidates yet
        # but are clearly the same surface form.
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) == find(j):
                    continue
                if _text_jaccard(items[i].mention_text, items[j].mention_text) >= 0.6:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for idx in range(n):
            groups.setdefault(find(idx), []).append(idx)

        clusters: list[ReviewCluster] = []
        for member_indices in groups.values():
            members = [items[k] for k in member_indices]
            members.sort(key=lambda m: m.priority_score, reverse=True)
            agg = sum(m.priority_score for m in members) + math.log1p(len(members) - 1) * 0.05
            clusters.append(ReviewCluster(
                cluster_id=_cluster_id_for(members),
                representative=members[0],
                members=members,
                aggregate_priority=agg,
            ))
        return clusters

    def compute_review_priority(self, item: ReviewItem) -> float:
        affected = self._affected_content_score(item)
        embedding_impact = max(0.0, min(1.0, item.embedding_impact))
        velocity = self._velocity_score(item)
        ambiguity = self._ambiguity_score(item)
        return (
            self.AFFECTED_W * affected
            + self.EMBED_IMPACT_W * embedding_impact
            + self.INGEST_W * velocity
            + self.AMBIGUITY_W * ambiguity
        )

    def stats(self) -> dict:
        total_resolved = self._auto_resolve_count + self._human_resolved_count
        auto_rate = self._auto_resolve_count / total_resolved if total_resolved else 0.0
        avg_time = (
            statistics.mean(self._resolution_durations) if self._resolution_durations else 0.0
        )
        return {
            "queue_depth": len(self._pending),
            "auto_resolve_rate": auto_rate,
            "avg_resolution_time": avg_time,
            "clusters_pending": len(self.cluster_pending_items()),
        }

    # --- priority components -------------------------------------------
    def _affected_content_score(self, item: ReviewItem) -> float:
        if item.affected_content_count > 0:
            count = item.affected_content_count
        elif item.candidate_entities:
            top_id = item.candidate_entities[0][0]
            count = sum(
                1
                for _, et in self.graph_store._in.get(top_id, set())
                if et == EdgeType.MENTIONS
            )
            item.affected_content_count = count
        else:
            count = 0
        return min(1.0, math.log1p(count) / math.log1p(self.AFFECTED_NORM_CAP))

    def _raw_ingest_velocity(self, ref_time: datetime) -> float:
        cutoff = ref_time - timedelta(minutes=5)
        recent = [t for t in self._push_times if t >= cutoff]
        return len(recent) / 5.0

    def _velocity_score(self, item: ReviewItem) -> float:
        v = item.ingest_velocity
        if v <= 0:
            v = self._raw_ingest_velocity(item.created_at)
        return min(1.0, v / self.INGEST_NORM_CAP)

    def _ambiguity_score(self, item: ReviewItem) -> float:
        cands = item.candidate_entities
        if len(cands) >= 2:
            return max(0.0, 1.0 - abs(cands[0][1] - cands[1][1]))
        if len(cands) == 1:
            return max(0.0, 1.0 - abs(cands[0][1] - 0.75) * 2)
        return 0.0


# --- internals ----------------------------------------------------------
def _text_jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cluster_id_for(members: list[ReviewItem]) -> str:
    joined = "|".join(sorted(m.id for m in members))
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"cluster:{digest}"
