"""Grad-CAM explanations for the grading model, via jacobgil/pytorch-grad-cam.

The CAM's native grid size (before upsampling to the input resolution) is
recorded and exposed explicitly -- honesty about this resolution limit is
part of the deliverable (see AGENTS.md: Grad-CAM at 14x14 cannot localise
15-pixel microaneurysms).
"""
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from ..grading.model import GradingModel
from ..grading.ordinal import referable_score

_METHODS = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus}


class _ReferableTarget:
    """Grad-CAM target: backprop through P(grade >= 2), the referable-DR score."""

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
        return referable_score(model_output).squeeze()


def _tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Undo this project's (x-0.5)/0.5 normalization; return HxWx3 uint8 RGB."""
    img = (tensor.detach().cpu().numpy() * 0.5 + 0.5) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def compute_gradcam(model: GradingModel, image_tensor: torch.Tensor, method: str = "gradcam") -> dict:
    """Grad-CAM explanation for `model`'s referable-DR decision on one
    preprocessed image tensor (shape [1, 3, H, W]).

    Returns:
        {
          "cam_native": CAM downsampled to the target layer's native grid
            (an approximation of the true pre-upsampling CAM, not a
            re-derivation from raw activations -- see module code comment),
          "native_size": (h, w) of that native grid, e.g. (14, 14),
          "cam_upsampled": CAM resized to the input image's H, W,
          "overlay": input image with the CAM overlaid (RGB uint8),
        }
    """
    if method not in _METHODS:
        raise ValueError(f"unknown Grad-CAM method: {method!r}, choose from {list(_METHODS)}")

    target_layer = model.target_layer()

    native_shape = {}

    def _capture_native_shape(_module, _input, output):
        native_shape["hw"] = tuple(output.shape[-2:])

    handle = target_layer.register_forward_hook(_capture_native_shape)
    try:
        cam_algorithm = _METHODS[method](model=model, target_layers=[target_layer])
        grayscale_cam = cam_algorithm(input_tensor=image_tensor, targets=[_ReferableTarget()])[0]
    finally:
        handle.remove()

    native_h, native_w = native_shape.get("hw", grayscale_cam.shape)
    # Downsampling the already-upsampled CAM back to native size loses some
    # precision relative to re-deriving it from raw activations/gradients,
    # but avoids depending on this library's internal (version-fragile) CAM
    # math -- adequate for reporting "here is roughly the native grid the
    # model reasoned at," which is the honesty requirement this exists for.
    cam_native = cv2.resize(grayscale_cam, (native_w, native_h), interpolation=cv2.INTER_AREA)

    image_uint8 = _tensor_to_uint8(image_tensor[0])
    overlay = show_cam_on_image(image_uint8.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)

    return {
        "cam_native": cam_native,
        "native_size": (int(native_h), int(native_w)),
        "cam_upsampled": grayscale_cam,
        "overlay": overlay,
    }


if __name__ == "__main__":
    import time

    from ..device import get_device

    t0 = time.time()
    device = get_device()
    model = GradingModel(backbone_name="resnet34", pretrained=False).to(device)
    model.eval()

    for method in ("gradcam", "gradcam++"):
        image_tensor = torch.randn(1, 3, 224, 224, device=device)
        result = compute_gradcam(model, image_tensor, method=method)
        assert result["cam_upsampled"].shape == (224, 224)
        assert result["overlay"].shape == (224, 224, 3)
        nh, nw = result["native_size"]
        assert 1 <= nh <= 224 and 1 <= nw <= 224
        assert result["cam_native"].shape == (nh, nw)

    elapsed = time.time() - t0
    print(f"gradcam.py smoke test ok in {elapsed:.3f}s on {device}, native_size={result['native_size']}")
    assert elapsed < 30
