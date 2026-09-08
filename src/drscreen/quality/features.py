"""Handcrafted image-quality features, concatenated into the IQA CNN's head."""
import cv2
import numpy as np
from skimage.filters import frangi


def laplacian_variance(img: np.ndarray, mask: np.ndarray) -> float:
    """Focus proxy: variance of the Laplacian inside the FOV."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[mask].var()) if mask.any() else 0.0


def tenengrad(img: np.ndarray, mask: np.ndarray) -> float:
    """Focus proxy: mean squared Sobel gradient magnitude inside the FOV."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_sq = gx**2 + gy**2
    return float(grad_mag_sq[mask].mean()) if mask.any() else 0.0


def illumination_uniformity(img: np.ndarray, mask: np.ndarray, block_size: int = 32) -> float:
    """Std of block-wise mean brightness across the FOV. Lower = more uniform."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    block_means = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block_mask = mask[y : y + block_size, x : x + block_size]
            if block_mask.sum() < 0.5 * block_mask.size:
                continue
            block = gray[y : y + block_size, x : x + block_size]
            block_means.append(block[block_mask].mean())
    if len(block_means) < 2:
        return 0.0
    return float(np.std(block_means))


def fov_coverage(mask: np.ndarray) -> float:
    """Fraction of the frame that is retina."""
    return float(mask.mean())


def vessel_visibility(img: np.ndarray, mask: np.ndarray) -> float:
    """Mean Frangi vesselness response inside the FOV -- a strong quality proxy."""
    green = img[:, :, 1].astype(np.float64) / 255.0
    response = frangi(green, sigmas=range(1, 4))
    return float(response[mask].mean()) if mask.any() else 0.0


def contrast(img: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(gray[mask].std()) if mask.any() else 0.0


def saturation_clipping(img: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of in-FOV pixels clipped at the black or white end in any channel."""
    in_fov = img[mask]
    if in_fov.size == 0:
        return 0.0
    clipped = (in_fov <= 5) | (in_fov >= 250)
    return float(clipped.any(axis=-1).mean())


FEATURE_NAMES = [
    "laplacian_variance",
    "tenengrad",
    "illumination_uniformity",
    "fov_coverage",
    "vessel_visibility",
    "contrast",
    "saturation_clipping",
]


def extract_features(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fixed-length handcrafted quality feature vector, in `FEATURE_NAMES` order."""
    values = [
        laplacian_variance(img, mask),
        tenengrad(img, mask),
        illumination_uniformity(img, mask),
        fov_coverage(mask),
        vessel_visibility(img, mask),
        contrast(img, mask),
        saturation_clipping(img, mask),
    ]
    return np.array(values, dtype=np.float32)


if __name__ == "__main__":
    import time

    from ..preprocess.retina import cached_fov_mask

    rng = np.random.default_rng(2)
    mask = cached_fov_mask(448)
    t0 = time.time()
    for _ in range(5):
        img = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)
        feats = extract_features(img, mask)
        assert feats.shape == (len(FEATURE_NAMES),)
        assert not np.isnan(feats).any()
    elapsed = time.time() - t0
    print(f"features.py smoke test ok in {elapsed:.3f}s ({elapsed/5:.3f}s/image)")
    assert elapsed < 30
