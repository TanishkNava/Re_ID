import numpy as np
import yaml
from collections import defaultdict


class ReIDMemory:
    """
    Gallery that stores embeddings per person and matches
    new embeddings using cosine similarity.
    Assigns persistent person_1, person_2, ... IDs.
    """

    def __init__(self, config_path="configs/pipeline_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["memory"]

        self.cos_threshold  = cfg["cosine_threshold"]
        self.max_gallery    = cfg["max_gallery_size"]
        self.feature_dim    = cfg["feature_dim"]

        # person_id (int) → list of embeddings (keep last N)
        self.gallery        = {}
        self.next_id        = 1
        self.max_feats      = 10   # embeddings to keep per person

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _mean_embedding(self, person_id: int) -> np.ndarray:
        feats = np.array(self.gallery[person_id])
        return feats.mean(axis=0)

    def match(self, embedding: np.ndarray) -> int:
        """
        Match embedding against gallery.
        Returns existing person_id if match found, else assigns new id.
        """
        if embedding is None:
            return -1

        best_id    = -1
        best_score = -1.0

        for pid in self.gallery:
            mean_feat = self._mean_embedding(pid)
            score     = self._cosine_similarity(embedding, mean_feat)
            if score > best_score:
                best_score = score
                best_id    = pid

        if best_score >= self.cos_threshold:
            # Update gallery with new embedding
            self.gallery[best_id].append(embedding)
            if len(self.gallery[best_id]) > self.max_feats:
                self.gallery[best_id].pop(0)
            return best_id
        else:
            # New person
            new_id = self.next_id
            self.next_id += 1
            self.gallery[new_id] = [embedding]
            return new_id

    def match_batch(self, embeddings: list) -> list:
        """Match a list of embeddings, return list of person_ids."""
        return [self.match(e) for e in embeddings]

    def get_label(self, person_id: int) -> str:
        return f"person_{person_id}" if person_id > 0 else "person_?"

    def reset(self):
        self.gallery  = {}
        self.next_id  = 1