import numpy as np
import yaml
import cv2
from pathlib import Path
from core.detector import YOLOXDetector
from core.tracker import TrackerManager
from core.reid import OSNetReID
from core.memory import ReIDMemory

# Colour palette (BGR) for person IDs — up to 20 distinct colours
_PERSON_COLOURS = [
    (0, 255, 0),    # green
    (0, 200, 255),  # yellow-cyan
    (255, 128, 0),  # blue-orange
    (0, 255, 200),  # mint
    (200, 0, 255),  # magenta
    (0, 128, 255),  # orange
    (128, 255, 0),  # lime
    (255, 0, 128),  # pink
    (0, 80, 255),   # red-ish
    (255, 255, 0),  # cyan
]


def _person_colour(person_id: int):
    if person_id <= 0:
        return (128, 128, 128)
    return _PERSON_COLOURS[(person_id - 1) % len(_PERSON_COLOURS)]


class Pipeline:
    """
    Per-frame orchestrator.
    Connects Detector → Tracker → ReID → Memory into one call.

    Changes vs v1:
    - Passes track_id into memory.match() for stable track→person bridging
    - Calls memory.update_active_tracks() each frame to purge dead entries
    - Better drawing: unique colour per person, cleaner labels
    """

    def __init__(self, config_path="configs/pipeline_config.yaml"):
        # Resolve config relative to repo root so subprocess cd doesn't break it
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parent.parent / config_path

        print("[Pipeline] Initializing components...")
        self.detector = YOLOXDetector(str(config_path))
        self.tracker  = TrackerManager(str(config_path))
        self.reid     = OSNetReID(str(config_path))
        self.memory   = ReIDMemory(str(config_path))
        self._config_path = str(config_path)
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
        # ── 1. Detect ──────────────────────────────────────────────────
        person_dets, object_dets = self.detector.detect(frame)

        # ── 2a. Track persons ──────────────────────────────────────────
        person_tracks = self.tracker.person_tracker.update(person_dets, frame)

        # ── 2b. Track objects (Motion+IoU only) ────────────────────────
        object_tracks = self.tracker.object_tracker.update(object_dets, frame)

        # ── 3. Begin new frame in memory (resets uniqueness set + purges dead tracks)
        active_track_ids = {t["track_id"] for t in person_tracks}
        self.memory.begin_frame(active_track_ids)

        # ── 4. Re-ID on person crops ────────────────────────────────────
        person_results = []
        for track in person_tracks:
            crop      = track.get("crop")
            embedding = self.reid.extract(crop) if crop is not None else None
            person_id = self.memory.match(embedding, track_id=track["track_id"])
            label     = self.memory.get_label(person_id)

            person_results.append({
                "track_id" : track["track_id"],
                "person_id": person_id,
                "label"    : label,
                "bbox"     : track["bbox"],
                "conf"     : track["conf"],
            })

        # ── 5. Format object results ────────────────────────────────────
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
        Persons → unique colour per person_id.
        Objects → blue box with class label.
        """
        class_names = self.detector.class_names

        BLUE  = (255, 100, 0)
        WHITE = (255, 255, 255)

        for p in result["persons"]:
            x1, y1, x2, y2 = p["bbox"]
            colour = _person_colour(p["person_id"])
            label  = f"{p['label']}  {p['conf']:.0%}"

            # Draw filled semi-transparent background for label
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        for o in result["objects"]:
            x1, y1, x2, y2 = o["bbox"]
            cid   = o["class_id"]
            cname = class_names[cid] if isinstance(cid, int) and cid < len(class_names) else "object"
            label = f"{cname} #{o['track_id']}  {o['conf']:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), BLUE, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        return frame