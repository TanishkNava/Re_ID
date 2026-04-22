import numpy as np
from scipy.optimize import linear_sum_assignment

class ReIDGallery:
    """
    Maintains a dictionary of recent disconnected/lost track embeddings to enable Re-Identification
    when they reappear. Old entries are cleared using a TTL.
    """
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        # Key: track_id, Value: {'embedding': np.array, 'frame_lost': int}
        self.gallery = {} 

    def update(self, lost_tracks_dict, current_frame):
        """
        lost_tracks_dict: {track_id: normalized_embedding (np.ndarray)}
        """
        # Expire out-of-date entries
        keys_to_remove = [tid for tid, data in self.gallery.items() 
                          if (current_frame - data['frame_lost']) > self.ttl]
        
        for tid in keys_to_remove:
            del self.gallery[tid]
            
        # Add or update new lost tracks
        for tid, emb in lost_tracks_dict.items():
            self.gallery[tid] = {'embedding': emb, 'frame_lost': current_frame}

    def query(self, features, min_similarity=0.65):
        """
        Compare active un-matched detection features against the gallery.
        Returns a list of matching track_ids (same length as features), filling None for no match.
        """
        if not self.gallery or len(features) == 0:
            return [None] * len(features)
        
        gallery_tids = list(self.gallery.keys())
        gallery_embs = np.array([self.gallery[tid]['embedding'] for tid in gallery_tids])
        
        # Cosine similarity. both are L2 normalized, so dot product is equivalent.
        sim_matrix = np.dot(features, gallery_embs.T) 
        
        # We need to maximize similarity, so cost is 1 - similarity.
        cost_matrix = 1.0 - sim_matrix
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = [None] * len(features)
        
        matched_gallery_tids = set()
        
        for r, c in zip(row_ind, col_ind):
            similarity = sim_matrix[r, c]
            if similarity >= min_similarity:
                matched_tid = gallery_tids[c]
                matches[r] = matched_tid
                matched_gallery_tids.add(matched_tid)
                
        # Remove matched templates so they aren't assigned to multiple future detections simultaneously
        # or retained uselessly.
        for tid in matched_gallery_tids:
            del self.gallery[tid]
            
        return matches
        
    def remove(self, track_id):
        """ Hard-remove from gallery (e.g., if matched earlier in logic) """
        if track_id in self.gallery:
            del self.gallery[track_id]
