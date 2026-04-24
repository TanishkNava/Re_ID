import numpy as np
import yaml
from core.detector import YOLOXDetector
from core.tracker import TrackerManager
from core.reid import OSNetReID
from core.memory import ReIDMemory


class Pipeline:
    """
    Per-frame orchestrator.
    Connects Detector → Tracker → ReID → Memory into one call.
    """

    def __init__(self, config_path="configs/pipeline_config.yaml"):
        print("[Pipeline] Initializing components...")
        self.detector = YOLOXDetector(config_path)
        self.tracker  = TrackerManager(config_path)
        self.reid     = OSNetReID(config_path)
        self.memory   = ReIDMemory(config_path)
        print("[Pipeline] Ready.")

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Args:
            frame: BGR numpy array
        Returns:
            result: {
                "persons"     : [{track_id, person_id, label, bbox, conf}],
                "objects"     : [{track_id, class_id, bbox, conf}],
            }
        """
        # ── 1. Detect ──────────────────────────────────────────────
        person_dets, object_dets = self.detector.detect(frame)

        print(f"[DEBUG] Frame detections: {len(person_dets)} persons, {len(object_dets)} objects")
        if person_dets:
            print(f"[DEBUG] Person detections: {person_dets}")
        # ── 2a. Track persons → get crops ──────────────────────────
        person_tracks = self.tracker.person_tracker.update(person_dets, frame)

        # ── 2b. Track objects (Motion+IoU only) ────────────────────
        object_tracks = self.tracker.object_tracker.update(object_dets, frame)

        # ── 3. Re-ID on person crops ────────────────────────────────
        person_results = []
        for track in person_tracks:
            crop      = track.get("crop")
            embedding = self.reid.extract(crop) if crop is not None else None
            person_id = self.memory.match(embedding)
            label     = self.memory.get_label(person_id)

            person_results.append({
                "track_id" : track["track_id"],
                "person_id": person_id,
                "label"    : label,
                "bbox"     : track["bbox"],
                "conf"     : track["conf"],
            })

        # ── 4. Format object results ────────────────────────────────
        object_results = []
        for track in object_tracks:
            object_results.append({
                "track_id" : track["track_id"],
                "class_id" : track["class_id"],
                "bbox"     : track["bbox"],
                "conf"     : track["conf"],
            })

        return {
            "persons": person_results,
            "objects": object_results,
        }

    def draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw bboxes and labels onto frame.
        Persons → green box with person_N label.
        Objects → blue box with class label.
        """
        import cv2

        GREEN = (0, 255, 0)
        BLUE  = (255, 100, 0)
        WHITE = (255, 255, 255)

        for p in result["persons"]:
            x1, y1, x2, y2 = p["bbox"]
            label = f"{p['label']} ({p['conf']:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

        from core.detector import YOLOXDetector
        # reuse class names from detector config
        import yaml
        with open("configs/pipeline_config.yaml") as f:
            class_names = yaml.safe_load(f)["detector"]["class_names"]

        for o in result["objects"]:
            x1, y1, x2, y2 = o["bbox"]
            cid   = o["class_id"]
            cname = class_names[cid] if isinstance(cid, int) and cid < len(class_names) else "object"
            label = f"{cname} t{o['track_id']} ({o['conf']:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)

        return frame