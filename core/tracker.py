import numpy as np
import yaml

from core.bytetrack.byte_tracker import BYTETracker


class TrackerArgs:
    """Minimal args object ByteTrack expects."""
    def __init__(self, cfg):
        self.track_thresh = cfg["track_thresh"]
        self.track_buffer = cfg["track_buffer"]
        self.match_thresh = cfg["match_thresh"]
        self.mot20        = False


class PersonTracker:
    """ByteTrack for persons — uses Motion + IoU + Re-ID handoff."""
    def __init__(self, cfg):
        args = TrackerArgs(cfg)
        self.tracker = BYTETracker(args, frame_rate=cfg["frame_rate"])

    def update(self, detections: list, frame: np.ndarray):
        """
        Args:
            detections: list of {bbox:[x1,y1,x2,y2], conf:float, class_id:int}
            frame:      original BGR frame
        Returns:
            tracks: list of {track_id, bbox, conf, class_id, crop}
        """
        if not detections:
            return []

        dets_array = np.array([
            [*d["bbox"], d["conf"], 0] for d in detections  # Force class_id=0 for person
        ], dtype=float)

        img_h, img_w = frame.shape[:2]
        online_targets = self.tracker.update(dets_array, [img_h, img_w], [img_h, img_w])

        tracks = []
        for t in online_targets:
            x1, y1, w, h = t.tlwh
            x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
            # clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            # Ensure crop is valid
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame[y1:y2, x1:x2].copy()

            tracks.append({
                "track_id" : int(t.track_id),
                "bbox"     : [x1, y1, x2, y2],
                "conf"     : float(t.score),
                "class_id" : 0,   # person class
                "crop"     : crop
            })
        return tracks


class ObjectTracker:
    """ByteTrack for non-persons — Motion + IoU only, no Re-ID."""
    def __init__(self, cfg):
        args = TrackerArgs(cfg)
        self.tracker = BYTETracker(args, frame_rate=cfg["frame_rate"])

    def update(self, detections: list, frame: np.ndarray):
        """
        Args:
            detections: list of {bbox:[x1,y1,x2,y2], conf:float, class_id:int}
            frame:      original BGR frame
        Returns:
            tracks: list of {track_id, bbox, conf, class_id}
        """
        if not detections:
            return []

        # Preserve original class_ids
        class_id_map = {i: d["class_id"] for i, d in enumerate(detections)}
        
        dets_array = np.array([
            [*d["bbox"], d["conf"], i] for i, d in enumerate(detections)  # Use index as temp ID
        ], dtype=float)

        img_h, img_w = frame.shape[:2]
        online_targets = self.tracker.update(dets_array, [img_h, img_w], [img_h, img_w])

        tracks = []
        for t in online_targets:
            x1, y1, w, h = t.tlwh
            x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
            # clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            # Get original class_id from detection index
            det_idx = int(t.cls) if hasattr(t, 'cls') else 0
            original_class_id = class_id_map.get(det_idx, -1)

            tracks.append({
                "track_id" : int(t.track_id),
                "bbox"     : [x1, y1, x2, y2],
                "conf"     : float(t.score),
                "class_id" : original_class_id,
            })
        return tracks


class TrackerManager:
    """Holds both trackers, loads config once."""
    def __init__(self, config_path="configs/pipeline_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["tracker"]
        self.person_tracker = PersonTracker(cfg)
        self.object_tracker = ObjectTracker(cfg)