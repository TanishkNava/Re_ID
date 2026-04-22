import numpy as np
import cv2
import warnings

class ReIDExtractor:
    def __init__(self, model_name='osnet_x1_0', device='cuda'):
        self.device = device
        self.use_fallback = False
        try:
            from torchreid.utils import FeatureExtractor
            self.extractor = FeatureExtractor(
                model_name=model_name,
                model_path='',
                device=self.device,
                image_size=(256, 128)
            )
            print("Successfully loaded torchreid OSNet!")
        except ImportError:
            print("Warning: torchreid not installed. Falling back to native PyTorch ResNet18 Re-ID extractor for demonstration...")
            self.use_fallback = True
            self._init_fallback()

    def _init_fallback(self):
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Strip final FC layer to get 512-dim features
        self.fallback_model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.fallback_model = self.fallback_model.to(self.device).eval()

    def extract(self, img_crops_bgr):
        if not img_crops_bgr:
            return np.empty((0, 512))
            
        img_crops_rgb = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in img_crops_bgr]
        
        if self.use_fallback:
            import torch
            tensors = []
            for img in img_crops_rgb:
                tensors.append(self.transform(img))
            batch = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                features = self.fallback_model(batch)
                features = features.view(features.size(0), -1) # Flatten to (N, 512)
            features = features.cpu().numpy()
        else:
            features = self.extractor(img_crops_rgb)
            features = features.cpu().numpy()
            
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        features = features / norms
        
        return features
