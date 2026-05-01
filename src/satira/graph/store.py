from datetime import datetime
from typing import Any

from satira.graph.schema import (
    ContentNode,
    EdgeType,
    EntityNode,
    EventNode,
    SourceNode,
    TemplateNode,
)


class GraphStore:
    """In-memory graph store with versioned snapshots.

    Uses dicts for fast lookup. Edges are stored as forward and reverse
    adjacency sets so neighbor queries and entity merges are O(degree).
    Snapshots return a JSON-serializable dict (datetimes become ISO
    strings) suitable for persistence.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Any] = {}
        self._node_kind: dict[str, str] = {}
        self._out: dict[str, set[tuple[str, EdgeType]]] = {}
        self._in: dict[str, set[tuple[str, EdgeType]]] = {}

    # --- node insertion -------------------------------------------------
    def _add_node(self, node: Any, kind: str) -> None:
        if node.id in self._nodes:
            raise ValueError(f"node id {node.id!r} already exists")
        self._nodes[node.id] = node
        self._node_kind[node.id] = kind
        self._out[node.id] = set()
        self._in[node.id] = set()

    def add_entity(self, entity: EntityNode) -> None:
        self._add_node(entity, "entity")

    def add_source(self, source: SourceNode) -> None:
        self._add_node(source, "source")

    def add_event(self, event: EventNode) -> None:
        self._add_node(event, "event")

    def add_template(self, template: TemplateNode) -> None:
        self._add_node(template, "template")

    def add_content(self, content: ContentNode) -> None:
        self._add_node(content, "content")

    # --- edges ----------------------------------------------------------
    def add_edge(self, from_id: str, to_id: str, edge_type: EdgeType) -> None:
        if from_id not in self._nodes:
            raise KeyError(f"unknown source node {from_id!r}")
        if to_id not in self._nodes:
            raise KeyError(f"unknown target node {to_id!r}")
        self._out[from_id].add((to_id, edge_type))
        self._in[to_id].add((from_id, edge_type))

    # --- lookup ---------------------------------------------------------
    def get_entity(self, entity_id: str) -> EntityNode | None:
        node = self._nodes.get(entity_id)
        if isinstance(node, EntityNode):
            return node
        return None

    def get_neighbors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[str]:
        if node_id not in self._out:
            return []
        if edge_type is None:
            return [target for target, _ in self._out[node_id]]
        return [target for target, et in self._out[node_id] if et == edge_type]

    # --- merge ----------------------------------------------------------
    def merge_entities(self, source_id: str, target_id: str) -> list[str]:
        """Merge ``source_id`` into ``target_id``; redirect all edges.

        Returns sorted list of node IDs touched by the merge (the target
        plus every neighbor whose edge was rewritten).
        """
        if source_id == target_id:
            return [target_id]
        src = self.get_entity(source_id)
        tgt = self.get_entity(target_id)
        if src is None:
            raise KeyError(f"unknown entity {source_id!r}")
        if tgt is None:
            raise KeyError(f"unknown entity {target_id!r}")

        affected: set[str] = {target_id}

        # absorb source's canonical name + aliases into target's aliases
        merged = list(tgt.aliases)
        for alias in (src.canonical_name, *src.aliases):
            if alias and alias != tgt.canonical_name and alias not in merged:
                merged.append(alias)
        tgt.aliases = merged

        for to_id, et in list(self._out[source_id]):
            self._in[to_id].discard((source_id, et))
            if to_id == target_id:
                continue  # would become a self-loop after merge
            self._out[target_id].add((to_id, et))
            self._in[to_id].add((target_id, et))
            affected.add(to_id)

        for from_id, et in list(self._in[source_id]):
            self._out[from_id].discard((source_id, et))
            if from_id == target_id:
                continue
            self._in[target_id].add((from_id, et))
            self._out[from_id].add((target_id, et))
            affected.add(from_id)

        del self._nodes[source_id]
        del self._node_kind[source_id]
        del self._out[source_id]
        del self._in[source_id]

        return sorted(affected)

    # --- persistence ----------------------------------------------------
    def snapshot(self) -> dict:
        nodes = []
        for nid, node in self._nodes.items():
            kind = self._node_kind[nid]
            nodes.append({"kind": kind, "data": _NODE_SERIALIZERS[kind](node)})

        edges = []
        for from_id, out_set in self._out.items():
            for to_id, et in sorted(out_set, key=lambda pair: (pair[0], pair[1].value)):
                edges.append({"from": from_id, "to": to_id, "type": et.value})

        return {"nodes": nodes, "edges": edges}

    def restore(self, snapshot: dict) -> None:
        self._nodes = {}
        self._node_kind = {}
        self._out = {}
        self._in = {}
        for entry in snapshot.get("nodes", []):
            kind = entry["kind"]
            node = _NODE_DESERIALIZERS[kind](entry["data"])
            self._add_node(node, kind)
        for entry in snapshot.get("edges", []):
            self.add_edge(entry["from"], entry["to"], EdgeType(entry["type"]))


def _entity_to_dict(e: EntityNode) -> dict:
    return {
        "id": e.id,
        "canonical_name": e.canonical_name,
        "entity_type": e.entity_type,
        "aliases": list(e.aliases),
        "created_at": e.created_at.isoformat(),
    }


def _entity_from_dict(d: dict) -> EntityNode:
    return EntityNode(
        id=d["id"],
        canonical_name=d["canonical_name"],
        entity_type=d["entity_type"],
        aliases=list(d["aliases"]),
        created_at=datetime.fromisoformat(d["created_at"]),
    )


def _source_to_dict(s: SourceNode) -> dict:
    return {
        "id": s.id,
        "domain": s.domain,
        "account_id": s.account_id,
        "credibility_label": s.credibility_label,
    }


def _source_from_dict(d: dict) -> SourceNode:
    return SourceNode(
        id=d["id"],
        domain=d["domain"],
        account_id=d["account_id"],
        credibility_label=d["credibility_label"],
    )


def _event_to_dict(e: EventNode) -> dict:
    return {
        "id": e.id,
        "topic_cluster": e.topic_cluster,
        "date_range": [e.date_range[0].isoformat(), e.date_range[1].isoformat()],
    }


def _event_from_dict(d: dict) -> EventNode:
    start, end = d["date_range"]
    return EventNode(
        id=d["id"],
        topic_cluster=d["topic_cluster"],
        date_range=(datetime.fromisoformat(start), datetime.fromisoformat(end)),
    )


def _template_to_dict(t: TemplateNode) -> dict:
    return {
        "id": t.id,
        "perceptual_hash": t.perceptual_hash,
        "layout_features": list(t.layout_features),
    }


def _template_from_dict(d: dict) -> TemplateNode:
    return TemplateNode(
        id=d["id"],
        perceptual_hash=d["perceptual_hash"],
        layout_features=list(d["layout_features"]),
    )


def _content_to_dict(c: ContentNode) -> dict:
    return {
        "id": c.id,
        "image_hash": c.image_hash,
        "extracted_text": c.extracted_text,
        "timestamp": c.timestamp.isoformat(),
        "source_id": c.source_id,
    }


def _content_from_dict(d: dict) -> ContentNode:
    return ContentNode(
        id=d["id"],
        image_hash=d["image_hash"],
        extracted_text=d["extracted_text"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        source_id=d["source_id"],
    )


_NODE_SERIALIZERS = {
    "entity": _entity_to_dict,
    "source": _source_to_dict,
    "event": _event_to_dict,
    "template": _template_to_dict,
    "content": _content_to_dict,
}

_NODE_DESERIALIZERS = {
    "entity": _entity_from_dict,
    "source": _source_from_dict,
    "event": _event_from_dict,
    "template": _template_from_dict,
    "content": _content_from_dict,
}
