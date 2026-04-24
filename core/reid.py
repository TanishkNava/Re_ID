import torch
import numpy as np
import cv2
import yaml
import torchreid


class OSNetReID:
    def __init__(self, config_path="configs/pipeline_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["reid"]

        self.device     = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
        self.input_size = tuple(cfg["input_size"])  # (256, 128)

        # Build model
        self.model = torchreid.models.build_model(
            name=cfg["model_name"],
            num_classes=1000,
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[ReID] OSNet loaded on {self.device}")

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """BGR crop → normalized tensor (1, 3, H, W)."""
        img = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)                     # HWC → CHW
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Args:
            crop: BGR numpy array of a single person
        Returns:
            embedding: L2-normalised 512-d numpy vector
        """
        if crop is None or crop.size == 0:
            return None
        tensor = self.preprocess(crop)
        with torch.no_grad():
            feat = self.model(tensor)
        feat = feat.cpu().numpy().flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-8)    # L2 normalise
        return feat

    def extract_batch(self, crops: list) -> list:
        """Extract embeddings for a list of crops. Returns list of embeddings."""
        return [self.extract(c) for c in crops if c is not None]