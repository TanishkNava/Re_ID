import numpy as np
import yaml

from core.bytetrack.byte_tracker import BYTETracker


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    w, h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = w * h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-7)


def _best_det_class_for_bbox(bbox, detections: list) -> int:
    best_i, best_iou = -1, 0.0
    for i, d in enumerate(detections):
        iou = _iou_xyxy(bbox, d["bbox"])
        if iou > best_iou:
            best_iou, best_i = iou, i
    if best_i < 0:
        return -1
    return int(detections[best_i]["class_id"])


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

        # BYTETracker: shape (N,5) => score = col4. Shape (N,6+) => score = col4*col5 (YOLO obj×cls).
        # conf here is already combined [0,1]; do not pass a dummy 6th column or scores become 0.
        dets_array = np.array(
            [[*d["bbox"], d["conf"]] for d in detections],
            dtype=float,
        )

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

        dets_array = np.array(
            [[*d["bbox"], d["conf"]] for d in detections],
            dtype=float,
        )

        img_h, img_w = frame.shape[:2]
        online_targets = self.tracker.update(dets_array, [img_h, img_w], [img_h, img_w])

        tracks = []
        for t in online_targets:
            x1, y1, w, h = t.tlwh
            x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
            # clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            original_class_id = _best_det_class_for_bbox([x1, y1, x2, y2], detections)

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