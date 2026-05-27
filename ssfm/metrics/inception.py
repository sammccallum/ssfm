import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
except ImportError as e:
    raise ImportError(
        "torch and torchvision are required for FID computation. "
        "Install with: uv add --optional metrics torch torchvision"
    ) from e


class InceptionFeatureExtractor:
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = nn.Identity()  # pyright: ignore
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> np.ndarray:
        x = images.to(self.device)
        out = self.model(x)
        return out.cpu().numpy()
