import numpy as np
import cv2
import yaml

# #region agent log
from core.debug_log import log as _dbg_log
# #endregion

try:
    import torch
except Exception as e:
    torch = None  # type: ignore[assignment]
    # #region agent log
    _dbg_log(
        run_id="pre-fix",
        hypothesis_id="H2",
        location="core/reid.py:import",
        message="Failed to import torch",
        data={"error": repr(e)},
    )
    # #endregion

try:
    import torchreid  # type: ignore
except Exception as e:
    torchreid = None  # type: ignore[assignment]
    # #region agent log
    _dbg_log(
        run_id="pre-fix",
        hypothesis_id="H1",
        location="core/reid.py:import",
        message="torchreid import failed; will use fallback backend",
        data={"error": repr(e)},
    )
    # #endregion

try:
    from torchvision import models as tv_models  # type: ignore
except Exception as e:
    tv_models = None  # type: ignore[assignment]
    # #region agent log
    _dbg_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="core/reid.py:import",
        message="Failed to import torchvision models",
        data={"error": repr(e)},
    )
    # #endregion


class OSNetReID:
    def __init__(self, config_path="configs/pipeline_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["reid"]

        # #region agent log
        _dbg_log(
            run_id="pre-fix",
            hypothesis_id="H4",
            location="core/reid.py:OSNetReID.__init__",
            message="Initializing ReID",
            data={
                "config_path": str(config_path),
                "model_name": cfg.get("model_name"),
                "device_cfg": cfg.get("device"),
            },
        )
        # #endregion

        if torch is None:
            raise RuntimeError("PyTorch is required for ReID but could not be imported.")

        dev = str(cfg.get("device", "cuda")).strip().lower()
        if dev == "cpu":
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device(cfg.get("device", "cuda"))
        else:
            print("[ReID] CUDA requested but not available; using CPU")
            self.device = torch.device("cpu")
        self.input_size = tuple(cfg["input_size"])  # (256, 128)

        self.backend = "torchreid" if torchreid is not None else "torchvision_fallback"

        # Build model
        if torchreid is not None:
            self.model = torchreid.models.build_model(
                name=cfg["model_name"],
                num_classes=1000,
                pretrained=True
            )
        else:
            if tv_models is None:
                raise RuntimeError("Neither torchreid nor torchvision is available for ReID.")
            # Fallback: resnet18 embedding (512-d after avgpool). Lower accuracy than OSNet.
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            backbone.fc = torch.nn.Identity()
            self.model = backbone

        # #region agent log
        _dbg_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="core/reid.py:OSNetReID.__init__",
            message="ReID backend selected",
            data={"backend": self.backend, "device": str(self.device)},
        )
        # #endregion

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[ReID] ReID model loaded on {self.device} ({self.backend})")

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