import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanFilter
from .reid_extractor import ReIDExtractor
from .reid_gallery import ReIDGallery

def box_iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
        
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:,0], boxes1[:,1], boxes1[:,2], boxes1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:,0], boxes2[:,1], boxes2[:,2], boxes2[:,3]
    
    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)
    
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / np.maximum(union_area, 1e-7)
    return iou

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class Track:
    def __init__(self, bbox, score, class_id, feature, kf):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.score = score
        self.class_id = class_id
        self.feature = feature
        self.kf = kf
        
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w/2
        cy = bbox[1] + h/2
        self.mean, self.covariance = self.kf.initiate(np.array([cx, cy, w, h]))
        
        self.track_id = None
        self.is_activated = False
        self.state = TrackState.New
        
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.alpha = 0.9 # EMA weight
        
        self.status_out = "new"

    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        
        cx, cy, w, h = self.mean[:4]
        self.bbox = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    def update(self, det, frame_id, event="tracked"):
        self.time_since_update = 0
        self.hits += 1
        self.score = det.score
        
        if det.feature is not None:
            if self.feature is None:
                self.feature = det.feature
            else:
                self.feature = self.alpha * self.feature + (1 - self.alpha) * det.feature
                self.feature /= (np.linalg.norm(self.feature) + 1e-12)
                
        cx = det.bbox[0] + (det.bbox[2] - det.bbox[0]) / 2
        cy = det.bbox[1] + (det.bbox[3] - det.bbox[1]) / 2
        w, h = det.bbox[2]-det.bbox[0], det.bbox[3]-det.bbox[1]
        
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, np.array([cx, cy, w, h])
        )
        
        kx, ky, kw, kh = self.mean[:4]
        self.bbox = np.array([kx - kw/2, ky - kh/2, kx + kw/2, ky + kh/2])
        self.status_out = event

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

class Det:
    def __init__(self, bbox, conf, cls, feat=None):
        self.bbox = bbox
        self.score = conf
        self.class_id = cls
        self.feature = feat

class BotSortReID:
    def __init__(self, device='cuda', 
                 track_high_thresh=0.45, 
                 track_low_thresh=0.20, 
                 new_track_thresh=0.45, 
                 track_buffer=120, 
                 match_thresh=0.8,
                 min_hits=3,
                 reid_sim_thresh=0.65,
                 reid_gallery_ttl=300):
        self.device = device
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_hits = min_hits
        self.reid_sim_thresh = reid_sim_thresh
        
        self.extractor = ReIDExtractor(device=device)
        self.kf = KalmanFilter()
        
        self.tracks = []
        self.gallery = ReIDGallery(ttl=reid_gallery_ttl)
        self.next_id = 0
        self.frame_id = 0

    def reset(self):
        self.tracks = []
        self.gallery = ReIDGallery(ttl=self.gallery.ttl)
        self.next_id = 0
        self.frame_id = 0

    def _get_fused_cost(self, tracks, dets):
        if len(tracks) == 0 or len(dets) == 0:
            return np.empty((len(tracks), len(dets)))
            
        trk_boxes = np.array([t.bbox for t in tracks])
        det_boxes = np.array([d.bbox for d in dets])
        iou_matrix = box_iou(trk_boxes, det_boxes)
        iou_cost = 1.0 - iou_matrix
        
        trk_feats = np.array([t.feature for t in tracks])
        det_feats = np.array([d.feature for d in dets])
        cos_sim = np.dot(trk_feats, det_feats.T)
        reid_cost = 1.0 - cos_sim
        
        fused_cost = 0.5 * iou_cost + 0.5 * reid_cost
        fused_cost[iou_matrix < 0.1] = 1e5
        
        return fused_cost

    def _get_iou_cost(self, tracks, dets):
        if len(tracks) == 0 or len(dets) == 0:
            return np.empty((len(tracks), len(dets)))
            
        trk_boxes = np.array([t.bbox for t in tracks])
        det_boxes = np.array([d.bbox for d in dets])
        result = 1.0 - box_iou(trk_boxes, det_boxes)
        return result

    def _linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
            
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append((r, c))
                
        matched_tracks = [m[0] for m in matches]
        matched_dets = [m[1] for m in matches]
        
        u_track = [i for i in range(cost_matrix.shape[0]) if i not in matched_tracks]
        u_det = [i for i in range(cost_matrix.shape[1]) if i not in matched_dets]
        
        return matches, u_track, u_det

    def update(self, detections, frame):
        self.frame_id += 1
        
        # Only process specified tracked class (Person = 0)
        persons = detections[detections[:, 5] == 0] if len(detections) > 0 else np.empty((0, 6))
        
        # Segment arrays by confidence score
        high_mask = persons[:, 4] >= self.track_high_thresh
        low_mask = (persons[:, 4] >= self.track_low_thresh) & (persons[:, 4] < self.track_high_thresh)
        
        dets_high = persons[high_mask]
        dets_low = persons[low_mask]
        
        # Extract visual appearance features only for high conf bounding boxes
        features_high = []
        if len(dets_high) > 0:
            crops = []
            for x1, y1, x2, y2, conf, cls in dets_high:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                
                # Check for validity
                if x2 <= x1 or y2 <= y1:
                    crop = np.zeros((128, 64, 3), dtype=np.uint8)
                else:
                    crop = frame[y1:y2, x1:x2]
                crops.append(crop)
            features_high = self.extractor.extract(crops)
            
        dets_high_obj = [Det(dets_high[i, :4], dets_high[i, 4], dets_high[i, 5], features_high[i]) for i in range(len(dets_high))]
        dets_low_obj = [Det(dets_low[i, :4], dets_low[i, 4], dets_low[i, 5], None) for i in range(len(dets_low))]
        
        # Partition known tracks into their respective statuses
        tracked_stracks = [t for t in self.tracks if t.state == TrackState.Tracked]
        lost_stracks = [t for t in self.tracks if t.state == TrackState.Lost]
        unconfirmed = [t for t in self.tracks if t.state == TrackState.New]
        
        strack_pool = tracked_stracks + lost_stracks
        
        # Step 1: Predict positions via Kalman Filter
        for t in self.tracks:
            t.predict()
            
        # Step 2: Stage 1 association (High Conf Detections vs Active + Lost Tracks)
        dists_stage1 = self._get_fused_cost(strack_pool, dets_high_obj)
        matches_1, u_track_1, u_det_1 = self._linear_assignment(dists_stage1, thresh=self.match_thresh)
        
        for itracked, idet in matches_1:
            track = strack_pool[itracked]
            det = dets_high_obj[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, event="tracked")
            else:
                track.update(det, self.frame_id, event="reidentified")
                track.state = TrackState.Tracked
                self.gallery.remove(track.track_id)
                
        # Step 3: Stage 2 association (Low Conf Detections vs Unmatched Tracked Tracks)
        u_track_1_objs = [strack_pool[i] for i in u_track_1]
        r_tracked_stracks = [t for t in u_track_1_objs if t.state == TrackState.Tracked]
        r_lost_stracks = [t for t in u_track_1_objs if t.state == TrackState.Lost]
        
        dists_stage2 = self._get_iou_cost(r_tracked_stracks, dets_low_obj)
        matches_2, u_track_2, u_det_2 = self._linear_assignment(dists_stage2, thresh=0.5)
        
        for itracked, idet in matches_2:
            track = r_tracked_stracks[itracked]
            det = dets_low_obj[idet]
            track.update(det, self.frame_id, event="tracked")
            
        # Anything unmatched that was tracked, goes to Lost
        for it in u_track_2:
            track = r_tracked_stracks[it]
            track.mark_lost()
            
        # Unmatched lost tracks naturally age
        for track in r_lost_stracks:
            track.time_since_update += 1
            
        # Step 4: Validate unconfirmed tracks (Stage 3 handling unconfirmed)
        unmatched_high_dets = [dets_high_obj[i] for i in u_det_1]
        dists_stage3 = self._get_iou_cost(unconfirmed, unmatched_high_dets)
        matches_3, u_unconfirmed, u_det_3 = self._linear_assignment(dists_stage3, thresh=0.7)
        
        for itrack, idet in matches_3:
            track = unconfirmed[itrack]
            det = unmatched_high_dets[idet]
            track.update(det, self.frame_id, event="tracked")
            
        # Fail unconfirmed matches
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            
        # Step 5: Check Gallery for any fully new high-conf bounding box returns
        new_dets = [unmatched_high_dets[i] for i in u_det_3]
        if len(new_dets) > 0:
            new_feats = np.array([d.feature for d in new_dets])
            gallery_matches = self.gallery.query(new_feats, min_similarity=self.reid_sim_thresh)
            
            for idx, matched_tid in enumerate(gallery_matches):
                det = new_dets[idx]
                track = Track(det.bbox, det.score, det.class_id, det.feature, self.kf)
                
                if matched_tid is not None:
                    # Successful ReID Query
                    track.track_id = matched_tid
                    track.is_activated = True
                    track.state = TrackState.Tracked
                    track.hits = self.min_hits
                    track.status_out = "reidentified"
                else:
                    # Genuinely a brand new person 
                    track.is_activated = False
                    track.state = TrackState.New
                    
                self.tracks.append(track)
                
        # Handle logic for removing tracks that exceed life thresholds and add them to standard ReID Gallery
        lost_to_gallery = {}
        for track in self.tracks:
            if track.state == TrackState.Lost and track.time_since_update > self.track_buffer:
                track.mark_removed()
                if track.track_id is not None and track.is_activated:
                    lost_to_gallery[track.track_id] = track.feature
                    
        self.gallery.update(lost_to_gallery, self.frame_id)
        
        # Purge
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        
        # Promote any unconfirmed tracks that have reached standard hits
        for t in self.tracks:
            if t.state == TrackState.New and t.hits >= self.min_hits:
                t.is_activated = True
                t.state = TrackState.Tracked
                if t.track_id is None:
                    self.next_id += 1
                    t.track_id = self.next_id
                    t.status_out = "new"
                    
        # Filter return payload
        rendered_tracks = []
        for t in self.tracks:
            if t.is_activated and t.state == TrackState.Tracked:
                rendered_tracks.append({
                    "track_id": t.track_id,
                    "bbox": [float(v) for v in t.bbox],
                    "confidence": float(t.score),
                    "class_id": int(t.class_id),
                    "status": t.status_out
                })
                # Drop ephemeral state statuses out after they've been captured by render
                if t.status_out in ["new", "reidentified"]:
                    t.status_out = "tracked"
                    
        return rendered_tracks
