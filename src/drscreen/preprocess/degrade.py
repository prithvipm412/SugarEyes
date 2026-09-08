"""Synthetic fundus-camera degradations.

Serves two roles (see AGENTS.md): IQA training augmentation, and the no-camera
demo input path -- since this project has no physical fundus camera, this is
how field-condition variation gets introduced at all. Every use of this
module must be labeled as synthetic, never presented as real field data.
"""
import cv2
import numpy as np


def defocus_blur(img: np.ndarray, severity: float) -> np.ndarray:
    ksize = max(1, int(round(severity * 25)) | 1)
    if ksize <= 1:
        return img.copy()
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def motion_blur(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    ksize = max(1, int(round(severity * 21)) | 1)
    if ksize <= 1:
        return img.copy()
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    angle_deg = float(rng.uniform(0, 180))
    rot = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (ksize, ksize))
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel)


def uneven_illumination(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist /= dist.max() + 1e-8
    gain = np.clip(1.0 + severity * (0.6 - 1.2 * dist), 0.2, 1.6)[:, :, None]
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def haze(img: np.ndarray, severity: float) -> np.ndarray:
    veil = np.full_like(img, 255)
    alpha = severity * 0.6
    return cv2.addWeighted(img, 1 - alpha, veil, alpha, 0)


def dust_specks(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    n_specks = int(severity * 40)
    for _ in range(n_specks):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 4))
        gray = int(rng.uniform(0, 60))
        cv2.circle(out, (x, y), r, (gray, gray, gray), -1)
    return out


def exposure_shift(img: np.ndarray, severity: float, direction: str, rng: np.random.Generator) -> np.ndarray:
    sign = -1 if direction == "under" else 1
    shift = sign * severity * 120 * rng.uniform(0.7, 1.0)
    return np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)


def simulate_capture(img: np.ndarray, severity: float, seed: int) -> np.ndarray:
    """Compose a random realistic subset of degradations at the given severity.

    `severity` in [0, 1] scales how strong each chosen degradation is.
    Deterministic given (img, severity, seed).
    """
    rng = np.random.default_rng(seed)
    out = img.copy()
    if severity <= 0:
        return out

    candidates = [
        lambda im: defocus_blur(im, severity * rng.uniform(0.5, 1.0)),
        lambda im: motion_blur(im, severity * rng.uniform(0.3, 1.0), rng),
        lambda im: uneven_illumination(im, severity * rng.uniform(0.5, 1.0), rng),
        lambda im: haze(im, severity * rng.uniform(0.3, 1.0)),
        lambda im: dust_specks(im, severity * rng.uniform(0.5, 1.0), rng),
        lambda im: exposure_shift(im, severity * rng.uniform(0.3, 1.0), str(rng.choice(["under", "over"])), rng),
    ]
    n_apply = int(rng.integers(1, 4))
    chosen = rng.choice(len(candidates), size=min(n_apply, len(candidates)), replace=False)
    for idx in chosen:
        out = candidates[int(idx)](out)
    return out


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(1)
    t0 = time.time()
    for i in range(5):
        img = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)
        out = simulate_capture(img, severity=0.7, seed=i)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        # determinism: same (img, severity, seed) must reproduce exactly
        out2 = simulate_capture(img, severity=0.7, seed=i)
        assert np.array_equal(out, out2)
    elapsed = time.time() - t0
    print(f"degrade.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
