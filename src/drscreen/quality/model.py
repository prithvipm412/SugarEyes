"""IQA model: a small CNN (timm backbone) with handcrafted quality features
concatenated into the classifier head. 3 classes: GRADEABLE / ENHANCEABLE / REJECT."""
import numpy as np
import timm
import torch
import torch.nn as nn

from .features import FEATURE_NAMES

CLASS_NAMES = ["GRADEABLE", "ENHANCEABLE", "REJECT"]
N_HANDCRAFTED_FEATURES = len(FEATURE_NAMES)

# Which handcrafted feature is worst -> what to tell the user on REJECT.
REJECT_REASONS = {
    "laplacian_variance": "out of focus",
    "tenengrad": "out of focus",
    "illumination_uniformity": "uneven illumination",
    "fov_coverage": "partial field of view",
    "vessel_visibility": "insufficient detail to assess vessels",
    "contrast": "low contrast",
    "saturation_clipping": "overexposed or underexposed",
}

# Direction in which each feature indicates a *worse* image: -1 means lower
# values are worse, +1 means higher values are worse.
FEATURE_BAD_DIRECTION = {
    "laplacian_variance": -1,
    "tenengrad": -1,
    "illumination_uniformity": 1,
    "fov_coverage": -1,
    "vessel_visibility": -1,
    "contrast": -1,
    "saturation_clipping": 1,
}


def compute_feature_stats(feature_matrix: np.ndarray) -> dict:
    """Mean/std per handcrafted feature over GRADEABLE reference images --
    the distribution `reject_reason` compares a rejected image's features against."""
    return {
        "mean": feature_matrix.mean(axis=0).tolist(),
        "std": (feature_matrix.std(axis=0) + 1e-6).tolist(),
    }


def reject_reason(feature_vector: np.ndarray, feature_stats: dict) -> str:
    """Pick a recapture reason from whichever handcrafted feature is most
    abnormal, in its "bad" direction, relative to `feature_stats`."""
    mean = np.array(feature_stats["mean"])
    std = np.array(feature_stats["std"])
    z = (feature_vector - mean) / std
    badness = [FEATURE_BAD_DIRECTION[name] * z[i] for i, name in enumerate(FEATURE_NAMES)]
    worst_idx = int(np.argmax(badness))
    return REJECT_REASONS[FEATURE_NAMES[worst_idx]]


class IQAModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", n_features: int = N_HANDCRAFTED_FEATURES, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + n_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, len(CLASS_NAMES)),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        embed = self.backbone(image)
        combined = torch.cat([embed, features], dim=1)
        return self.classifier(combined)


if __name__ == "__main__":
    import time

    from ..device import get_device

    device = get_device()
    model = IQAModel(pretrained=False).to(device)
    t0 = time.time()
    for _ in range(5):
        image = torch.randn(1, 3, 224, 224, device=device)
        features = torch.randn(1, N_HANDCRAFTED_FEATURES, device=device)
        out = model(image, features)
        assert out.shape == (1, len(CLASS_NAMES))
    elapsed = time.time() - t0

    rng = np.random.default_rng(4)
    reference = rng.normal(loc=10.0, scale=1.0, size=(50, N_HANDCRAFTED_FEATURES))
    stats = compute_feature_stats(reference)
    bad_focus = reference.mean(axis=0).copy()
    bad_focus[0] = reference.mean(axis=0)[0] - 20  # laplacian_variance crashes -> blur
    reason = reject_reason(bad_focus, stats)
    assert reason == "out of focus", reason

    print(f"model.py smoke test ok in {elapsed:.3f}s on {device}")
    assert elapsed < 30
