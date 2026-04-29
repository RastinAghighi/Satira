import math

import torch

from satira.data.contradiction_generator import AdversarialContradictionGenerator


class _StubGenerator(AdversarialContradictionGenerator):
    def __init__(self, embeddings: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stub_embeddings = embeddings

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        assert len(texts) == self._stub_embeddings.shape[0]
        return self._stub_embeddings


def _unit_at_angle(s: float) -> list[float]:
    return [s, math.sqrt(max(0.0, 1.0 - s * s))]


def _make_pairs(n: int) -> list[dict]:
    return [{"image_path": f"/img/{i}.jpg", "text": f"text_{i}"} for i in range(n)]


def test_similarity_window_respected() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            _unit_at_angle(0.30),
            _unit_at_angle(0.55),
            _unit_at_angle(0.85),
        ],
        dtype=torch.float32,
    )
    pairs = _make_pairs(4)
    gen = _StubGenerator(embeddings=embeddings, min_similarity=0.4, topic_threshold=0.7)

    results = gen.generate_hard_contradictions(pairs)

    pair_zero_results = [r for r in results if r["original_text"] == "text_0"]
    assert len(pair_zero_results) == 1, (
        f"expected exactly one mismatch for pair 0, got {pair_zero_results}"
    )
    assert pair_zero_results[0]["text"] == "text_2", (
        "pair 0 should mismatch with text_2 (sim 0.55), not text_1 (0.30) or text_3 (0.85)"
    )


def test_self_pairs_excluded() -> None:
    embeddings = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    pairs = _make_pairs(1)
    gen = _StubGenerator(embeddings=embeddings, min_similarity=0.4, topic_threshold=1.0)

    results = gen.generate_hard_contradictions(pairs)

    assert results == [], (
        "single pair must not be paired with itself even when self-similarity is in window"
    )


def test_output_format_has_required_fields() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            _unit_at_angle(0.55),
        ],
        dtype=torch.float32,
    )
    pairs = [
        {"image_path": "/path/0.jpg", "text": "alpha"},
        {"image_path": "/path/1.jpg", "text": "beta"},
    ]
    gen = _StubGenerator(embeddings=embeddings, min_similarity=0.4, topic_threshold=0.7)

    results = gen.generate_hard_contradictions(pairs)

    required_keys = {"image_path", "text", "original_text", "label", "difficulty"}
    assert len(results) == 2
    for item in results:
        assert required_keys.issubset(item.keys()), f"missing keys in {item}"
        assert item["label"] == "synthetic_contradiction"
        assert item["difficulty"] == "hard"

    by_image = {item["image_path"]: item for item in results}
    assert by_image["/path/0.jpg"]["text"] == "beta"
    assert by_image["/path/0.jpg"]["original_text"] == "alpha"
    assert by_image["/path/1.jpg"]["text"] == "alpha"
    assert by_image["/path/1.jpg"]["original_text"] == "beta"


def test_no_valid_candidates_returns_empty() -> None:
    embeddings = torch.eye(3, dtype=torch.float32)
    pairs = _make_pairs(3)
    gen = _StubGenerator(embeddings=embeddings, min_similarity=0.4, topic_threshold=0.7)

    results = gen.generate_hard_contradictions(pairs)

    assert results == [], (
        f"orthogonal embeddings (sim=0) lie below the window — expected empty, got {results}"
    )


def test_all_too_similar_returns_empty() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            _unit_at_angle(0.95),
            _unit_at_angle(0.92),
        ],
        dtype=torch.float32,
    )
    pairs = _make_pairs(3)
    gen = _StubGenerator(embeddings=embeddings, min_similarity=0.4, topic_threshold=0.7)

    results = gen.generate_hard_contradictions(pairs)

    assert results == [], "near-duplicate embeddings exceed topic_threshold and must be skipped"
