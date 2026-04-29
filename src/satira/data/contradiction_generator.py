import torch
from torch import Tensor


class AdversarialContradictionGenerator:
    """Creates Tier 2 training data: topically coherent but semantically mismatched pairs.

    Random mispairing (Congress image + ice cream headline) is too easy. The
    model just learns topic mismatch, not semantic contradiction within a
    topic. This generator finds text pairs whose cosine similarity falls in
    the 0.4-0.7 range: same general subject, different specific claims, which
    mirrors real satire where image and text discuss the same topic but the
    claims don't match. The candidate closest to the midpoint of the window
    is the hardest mismatch and is preferred.
    """

    def __init__(
        self,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        topic_threshold: float = 0.7,
        min_similarity: float = 0.4,
    ) -> None:
        if min_similarity >= topic_threshold:
            raise ValueError(
                f"min_similarity ({min_similarity}) must be < topic_threshold ({topic_threshold})"
            )
        self.text_encoder_name = text_encoder_name
        self.topic_threshold = topic_threshold
        self.min_similarity = min_similarity
        self.target_similarity = (min_similarity + topic_threshold) / 2.0
        self._encoder = None

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.text_encoder_name)
        return self._encoder

    def _encode_texts(self, texts: list[str]) -> Tensor:
        encoder = self._load_encoder()
        return encoder.encode(texts, convert_to_tensor=True)

    def _compute_pairwise_similarities(self, embeddings: Tensor) -> Tensor:
        normalized = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        return normalized @ normalized.transpose(-1, -2)

    def generate_hard_contradictions(self, authentic_pairs: list[dict]) -> list[dict]:
        if not authentic_pairs:
            return []

        texts = [pair["text"] for pair in authentic_pairs]
        embeddings = self._encode_texts(texts)
        sims = self._compute_pairwise_similarities(embeddings)

        n = len(authentic_pairs)
        diag_mask = torch.eye(n, dtype=torch.bool, device=sims.device)
        in_window = (
            (sims >= self.min_similarity) & (sims <= self.topic_threshold) & ~diag_mask
        )

        distance = (sims - self.target_similarity).abs()
        distance = distance.masked_fill(~in_window, float("inf"))

        best_distances, best_idx = distance.min(dim=1)

        results: list[dict] = []
        for i in range(n):
            if not torch.isfinite(best_distances[i]):
                continue
            j = int(best_idx[i])
            results.append(
                {
                    "image_path": authentic_pairs[i]["image_path"],
                    "text": authentic_pairs[j]["text"],
                    "original_text": authentic_pairs[i]["text"],
                    "label": "synthetic_contradiction",
                    "difficulty": "hard",
                }
            )
        return results
