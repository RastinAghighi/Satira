"""Multi-track context synchronisation across GNN version upgrades.

When the offline pipeline ships a new GNN architecture, the embeddings
it produces live in a different coordinate system from the one the
production classifier was trained against — pairing them silently
would corrupt every downstream prediction. The synchroniser keeps a
*track* per GNN architecture and gates every deployment on the live
``CompatibilityMatrix`` so that only contexts the active model
actually understands ever touch production.

Two entry points drive the state machine:

  - ``on_new_context``: every 15 min the offline pipeline registers a
    fresh embedding snapshot. The synchroniser parks it on its track
    and, if the snapshot is compatible with the active model, makes it
    the live context. Incompatible snapshots stay parked so they're
    ready when a model that *can* read them gets promoted.
  - ``on_model_promoted``: when the deployment controller promotes a
    classifier checkpoint, the synchroniser switches to whichever
    parked context is the most recent on a track the new model
    accepts, then drops every track the new model can no longer read.

The synchroniser does not register models or contexts with the
``CompatibilityMatrix`` itself — that's the offline pipeline's job and
the deployment controller's job, respectively. It assumes both ends
have already done their bookkeeping by the time it is called, so it
can lean on ``check_compatibility`` and ``find_best_compatible_context``
without needing to carry training stats around.
"""
from __future__ import annotations

from satira.deployment.compatibility import CompatibilityMatrix


class MultiTrackSynchronizer:
    """Keeps per-GNN context tracks and gates deployments on compatibility.

    Each track maps a GNN architecture version to the most recent
    context_version that was registered on it. ``_active_model`` and
    ``_active_compatible_gnns`` capture which classifier is currently
    serving traffic; ``on_new_context`` consults them to decide whether
    a freshly arrived context is safe to swap in or has to wait for a
    compatible model to land.
    """

    def __init__(self, compatibility_matrix: CompatibilityMatrix) -> None:
        self._matrix = compatibility_matrix
        self._tracks: dict[str, str] = {}
        self._active_model: str | None = None
        self._active_compatible_gnns: tuple[str, ...] = ()
        self._active_context: str | None = None

    # --- entry points --------------------------------------------------
    def on_new_context(self, context_version: str, gnn_version: str) -> dict:
        """Park a freshly registered context on its track; deploy if compatible.

        The track is updated unconditionally — even an incompatible
        snapshot stays on its track so that when a future model with
        the matching GNN version is promoted, the synchroniser can pick
        it up immediately instead of waiting for the next 15-minute
        offline cycle.
        """
        self._tracks[gnn_version] = context_version

        if self._active_model is None:
            return {
                "deployed": False,
                "track": gnn_version,
                "reason": "no active model; context parked on track",
            }

        if gnn_version not in self._active_compatible_gnns:
            return {
                "deployed": False,
                "track": gnn_version,
                "reason": (
                    f"GNN architecture {gnn_version!r} not on active model "
                    f"{self._active_model!r}'s allow-list; context parked"
                ),
            }

        result = self._matrix.check_compatibility(
            self._active_model, context_version
        )
        if not result.compatible:
            return {
                "deployed": False,
                "track": gnn_version,
                "reason": result.reason,
            }

        self._active_context = context_version
        return {
            "deployed": True,
            "track": gnn_version,
            "reason": result.reason,
        }

    def on_model_promoted(
        self,
        model_version: str,
        compatible_gnn_versions: list[str],
    ) -> dict:
        """Adopt a newly promoted model, switch context, and drop dead tracks.

        Pruning happens *after* the context switch so the picked
        context is guaranteed to come from a track that survives.
        """
        self._active_model = model_version
        self._active_compatible_gnns = tuple(compatible_gnn_versions)

        chosen = self._matrix.find_best_compatible_context(model_version)
        self._active_context = chosen

        compatible_set = set(compatible_gnn_versions)
        pruned = [g for g in self._tracks if g not in compatible_set]
        for g in pruned:
            del self._tracks[g]

        return {
            "context_deployed": chosen,
            "old_tracks_pruned": pruned,
        }

    def get_active_tracks(self) -> dict[str, str]:
        """Return a copy of ``{gnn_version: latest_context_version}``."""
        return dict(self._tracks)
