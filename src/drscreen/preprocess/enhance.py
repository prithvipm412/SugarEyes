"""CLAHE, illumination normalization, and denoising. All mask-respecting:
pixels outside the FOV mask are never touched."""
import cv2
import numpy as np


def clahe_green(img: np.ndarray, mask: np.ndarray, clip_limit: float = 2.0, mode: str = "green") -> np.ndarray:
    """Apply CLAHE to either the green channel or L (of LAB), selectable via `mode`."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if mode == "green":
        out = img.copy()
        channel = img[:, :, 1]
        enhanced = clahe.apply(channel)
        out[:, :, 1] = np.where(mask, enhanced, channel)
        return out
    if mode == "lab":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        lab[:, :, 0] = np.where(mask, clahe.apply(l_channel), l_channel)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        for c in range(3):
            out[:, :, c] = np.where(mask, out[:, :, c], img[:, :, c])
        return out
    raise ValueError(f"unknown clahe mode: {mode}")


def illumination_normalize(img: np.ndarray, mask: np.ndarray, sigma_frac: float = 0.05) -> np.ndarray:
    """Ben Graham style: subtract a heavy Gaussian blur, then rescale to uint8."""
    h, w = img.shape[:2]
    sigma = max(1.0, sigma_frac * max(h, w))
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    diff = img.astype(np.float32) - blurred.astype(np.float32)
    normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8) * 255.0
    normalized = normalized.astype(np.uint8)
    out = img.copy()
    for c in range(img.shape[2]):
        out[:, :, c] = np.where(mask, normalized[:, :, c], img[:, :, c])
    return out


def denoise(img: np.ndarray, mask: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Denoise via bilateral filtering (fast) or non-local means (slower, higher quality)."""
    if method == "bilateral":
        denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    elif method == "nlm":
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
    else:
        raise ValueError(f"unknown denoise method: {method}")
    out = img.copy()
    for c in range(img.shape[2]):
        out[:, :, c] = np.where(mask, denoised[:, :, c], img[:, :, c])
    return out


if __name__ == "__main__":
    import time

    from .retina import cached_fov_mask

    rng = np.random.default_rng(0)
    mask = cached_fov_mask(448)
    t0 = time.time()
    for _ in range(5):
        img = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)
        for fn, kwargs in [
            (clahe_green, {}),
            (clahe_green, {"mode": "lab"}),
            (illumination_normalize, {}),
            (denoise, {}),
        ]:
            out = fn(img, mask, **kwargs)
            assert out.shape == img.shape
            assert np.array_equal(out[~mask], img[~mask]), f"{fn.__name__}{kwargs} touched outside the FOV mask"
    elapsed = time.time() - t0
    print(f"enhance.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
