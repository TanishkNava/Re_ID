import cv2
import torch
import numpy as np
import yaml
from pathlib import Path

# YOLOX imports
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import ValTransform


def _infer_num_classes_from_state_dict(state_dict: dict) -> int | None:
    """Read YOLOX head cls_pred channel count from a checkpoint (one value per detection class)."""
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if "cls_preds" in name and name.endswith(".weight") and tensor.ndim == 4:
            return int(tensor.shape[0])
    return None


class YOLOXDetector:
    def __init__(self, config_path="configs/pipeline_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["detector"]

        self.class_names   = list(cfg["class_names"])
        self.person_id     = cfg["person_class_id"]
        self.conf_thresh   = cfg["conf_threshold"]
        self.nms_thresh    = cfg["nms_threshold"]
        self.input_size    = tuple(cfg["input_size"])
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights_path = cfg["weights"]
        ckpt = None
        if Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            raw = ckpt.get("model", ckpt)
            if isinstance(raw, dict):
                nc_ckpt = _infer_num_classes_from_state_dict(raw)
                nc_yaml = len(self.class_names)
                if nc_ckpt is not None and nc_ckpt != nc_yaml:
                    print(
                        f"[Detector] Weights define num_classes={nc_ckpt}, "
                        f"config has {nc_yaml} names — using checkpoint count."
                    )
                    if nc_ckpt < nc_yaml:
                        dropped = self.class_names[nc_ckpt:]
                        self.class_names = self.class_names[:nc_ckpt]
                        print(f"[Detector] Dropped class name(s) (unused by this ckpt): {dropped}")
                    else:
                        for i in range(nc_yaml, nc_ckpt):
                            self.class_names.append(f"class_{i}")
                        print(
                            f"[Detector] Appended placeholder names class_{nc_yaml}..class_{nc_ckpt - 1}; "
                            f"set detector.class_names in YAML to match training order."
                        )
                if self.person_id >= len(self.class_names):
                    print(
                        f"[Detector] WARNING: person_class_id={self.person_id} is out of range "
                        f"for {len(self.class_names)} classes; clamping to 0."
                    )
                    self.person_id = 0

        # Build exp and model (num_classes must match checkpoint)
        exp = get_exp(None, cfg["model_name"])
        exp.num_classes    = len(self.class_names)
        exp.test_conf      = self.conf_thresh
        exp.nmsthre        = self.nms_thresh
        exp.test_size      = self.input_size

        self.model = exp.get_model().to(self.device)
        self.model.eval()

        if ckpt is not None:
            raw = ckpt.get("model", ckpt)
            self.model.load_state_dict(raw)
            print(f"[Detector] Weights loaded from {weights_path}")
        else:
            print(f"[Detector] WARNING: No weights at {weights_path}, using random weights")

        self.preproc = ValTransform(legacy=False)

    def detect(self, frame: np.ndarray):
        """
        Args:
            frame: BGR numpy array (H, W, 3)
        Returns:
            persons     : list of dicts {bbox, conf, class_id, class_name}
            non_persons : list of dicts {bbox, conf, class_id, class_name}
        """
        h, w = frame.shape[:2]
        img, _ = self.preproc(frame, None, self.input_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, len(self.class_names),
                self.conf_thresh, self.nms_thresh,
                class_agnostic=True
            )

        persons, non_persons = [], []
        if outputs[0] is None:
            return persons, non_persons

        detections = outputs[0].cpu().numpy()
        # Scale bboxes back to original frame size
        scale = min(self.input_size[0] / h, self.input_size[1] / w)

        for det in detections:
            x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det
            x1 = max(0, int(x1 / scale))
            y1 = max(0, int(y1 / scale))
            x2 = min(w, int(x2 / scale))
            y2 = min(h, int(y2 / scale))
            conf  = float(obj_conf * cls_conf)
            cls_id = int(cls_id)
            name  = self.class_names[cls_id] if cls_id < len(self.class_names) else "unknown"

            entry = {
                "bbox"      : [x1, y1, x2, y2],
                "conf"      : conf,
                "class_id"  : cls_id,
                "class_name": name
            }
            if cls_id == self.person_id:
                persons.append(entry)
            else:
                non_persons.append(entry)

        return persons, non_persons
