"""DR severity grading model: an ordinal (CORN) head over a timm backbone."""
import timm
import torch
import torch.nn as nn

from .ordinal import NUM_GRADES

# Last conv block name per backbone, for Grad-CAM targeting in Phase 4.
_LAST_CONV_BLOCK = {
    "resnet18": "layer4",
    "resnet34": "layer4",
    "resnet50": "layer4",
    "convnext_tiny": "stages.3",
}


class GradingModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet34", num_grades: int = NUM_GRADES, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_grades = num_grades
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_grades - 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image))

    def target_layer(self) -> nn.Module:
        """The last conv block, as an nn.Module -- what Grad-CAM (Phase 4) targets."""
        name = _LAST_CONV_BLOCK.get(self.backbone_name)
        if name is None:
            raise ValueError(f"No known last-conv-block name for backbone {self.backbone_name!r}; add one to _LAST_CONV_BLOCK")
        module = self.backbone
        for part in name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module


def export_onnx(model: GradingModel, out_path: str, image_size: int = 224, opset: int = 13) -> None:
    """Export to ONNX at the given opset. Uses the legacy TorchScript-based
    exporter (dynamo=False) -- PyTorch 2.14's newer dynamo exporter can't
    down-convert this graph to opset < 18 (an internal op needs axes as an
    attribute, not an input, in older opsets) and silently keeps opset 18
    instead. The legacy path natively hits the requested opset."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )


if __name__ == "__main__":
    import time

    from ..device import get_device

    device = get_device()
    model = GradingModel(pretrained=False).to(device)
    t0 = time.time()
    for _ in range(5):
        image = torch.randn(2, 3, 224, 224, device=device)
        logits = model(image)
        assert logits.shape == (2, NUM_GRADES - 1)
    target = model.target_layer()
    assert isinstance(target, nn.Module)
    elapsed = time.time() - t0
    print(f"model.py smoke test ok in {elapsed:.3f}s on {device}")
    assert elapsed < 30
