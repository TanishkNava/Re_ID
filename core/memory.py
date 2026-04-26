import time
import numpy as np
import yaml


class ReIDMemory:
    """
    Gallery that stores embeddings per person and matches
    new embeddings using cosine similarity.

    Key guarantees:
    - ONE person_id per frame maximum (frame-level uniqueness mutex)
      → person_3 cannot appear on 5 people simultaneously
    - Stable track→person_id bridge: ByteTrack track_id caches its
      person_id so we don't re-query the gallery every frame
    - TTL: gallery entries expire after `ttl_seconds` of inactivity
    - EMA embedding update for smooth gallery evolution
    """

    def __init__(self, config_path="configs/pipeline_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["memory"]

        self.cos_threshold = cfg["cosine_threshold"]
        self.max_gallery   = cfg["max_gallery_size"]
        self.feature_dim   = cfg["feature_dim"]
        self.ttl_seconds   = cfg.get("ttl_seconds", 120)
        self.ema_alpha     = cfg.get("ema_alpha", 0.3)
        self.max_feats     = cfg.get("max_feats_per_person", 10)

        # person_id → {"embedding": np.ndarray, "feats": list, "last_seen": float}
        self.gallery: dict[int, dict] = {}
        self.next_id = 1

        # ByteTrack track_id → person_id  (reset when track dies)
        self._track_to_person: dict[int, int] = {}
        # Tracks active in the previous frame (for cleanup)
        self._last_active_tracks: set[int] = set()

        # ── Frame-level uniqueness ───────────────────────────────────────
        # person_ids already assigned in the CURRENT frame.
        # Reset at the start of each new frame via begin_frame().
        self._frame_used_ids: set[int] = set()

    # ── Frame lifecycle ──────────────────────────────────────────────────

    def begin_frame(self, active_track_ids: set[int]):
        """
        Call ONCE at the start of processing every frame.
        - Resets the per-frame uniqueness set.
        - Purges track→person entries for dead tracks.
        """
        self._frame_used_ids = set()
        # Drop mappings for tracks that have disappeared
        gone = self._last_active_tracks - active_track_ids
        for tid in gone:
            self._track_to_person.pop(tid, None)
        self._last_active_tracks = set(active_track_ids)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _expire_gallery(self):
        """Remove gallery entries that haven't been seen within TTL."""
        now = time.time()
        expired = [pid for pid, v in self.gallery.items()
                   if now - v["last_seen"] > self.ttl_seconds]
        for pid in expired:
            del self.gallery[pid]
            print(f"[Memory] person_{pid} expired (TTL={self.ttl_seconds}s)")

    def _update_gallery(self, person_id: int, embedding: np.ndarray):
        """EMA-update the gallery embedding and timestamp."""
        entry = self.gallery[person_id]
        entry["embedding"] = (
            (1 - self.ema_alpha) * entry["embedding"]
            + self.ema_alpha * embedding
        )
        norm = np.linalg.norm(entry["embedding"])
        if norm > 1e-8:
            entry["embedding"] /= norm
        entry["feats"].append(embedding.copy())
        if len(entry["feats"]) > self.max_feats:
            entry["feats"].pop(0)
        entry["last_seen"] = time.time()

    # ── Public API ───────────────────────────────────────────────────────

    def match(self, embedding: np.ndarray, track_id: int | None = None) -> int:
        """
        Match embedding against gallery, enforcing frame-level uniqueness.

        Args:
            embedding: L2-normalised feature vector (may be None).
            track_id:  ByteTrack track_id for stable bridging.

        Returns:
            person_id (int ≥ 1) or -1 if embedding is None.
        """
        if embedding is None:
            return -1

        # ── Fast path: track already has a cached person_id ────────────
        if track_id is not None and track_id in self._track_to_person:
            pid = self._track_to_person[track_id]
            if pid in self.gallery and pid not in self._frame_used_ids:
                self._update_gallery(pid, embedding)
                self._frame_used_ids.add(pid)
                return pid
            elif pid in self._frame_used_ids:
                # Another track in this frame already claimed this id —
                # clear the cache and let re-matching decide.
                del self._track_to_person[track_id]

        # ── Expire stale gallery entries ────────────────────────────────
        self._expire_gallery()

        # ── Match against gallery (skip already-used ids this frame) ────
        best_id    = -1
        best_score = -1.0

        for pid, entry in self.gallery.items():
            if pid in self._frame_used_ids:
                continue  # ← CORE FIX: this person is already in the frame
            score = self._cosine_similarity(embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_id    = pid

        if best_score >= self.cos_threshold:
            self._update_gallery(best_id, embedding)
            assigned_id = best_id
        else:
            # New person — create gallery entry
            assigned_id = self.next_id
            self.next_id += 1
            self.gallery[assigned_id] = {
                "embedding": embedding.copy(),
                "feats":     [embedding.copy()],
                "last_seen": time.time(),
            }
            print(f"[Memory] New person: person_{assigned_id} "
                  f"(best_score={best_score:.3f} < threshold={self.cos_threshold})")

        # ── Register assignments ────────────────────────────────────────
        self._frame_used_ids.add(assigned_id)
        if track_id is not None:
            self._track_to_person[track_id] = assigned_id

        return assigned_id

    def match_batch(self, embeddings: list, track_ids: list | None = None) -> list:
        if track_ids is None:
            track_ids = [None] * len(embeddings)
        return [self.match(e, tid) for e, tid in zip(embeddings, track_ids)]

    def get_label(self, person_id: int) -> str:
        return f"person_{person_id}" if person_id > 0 else "person_?"

    def reset(self):
        self.gallery = {}
        self.next_id = 1
        self._track_to_person = {}
        self._last_active_tracks = set()
        self._frame_used_ids = set()