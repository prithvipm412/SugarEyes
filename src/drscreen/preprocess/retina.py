"""Retina field-of-view detection, cropping, and resizing."""
import cv2
import numpy as np


def detect_retina_circle(img: np.ndarray) -> tuple[tuple[int, int], int]:
    """Locate the circular fundus field of view via thresholding + largest contour.

    Args:
        img: RGB or grayscale uint8 image.

    Returns:
        ((cx, cy), radius) in pixel coordinates of `img`.

    Raises:
        ValueError: if no FOV contour is found (e.g. an all-black image).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No retina FOV contour found in image")
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return (int(round(cx)), int(round(cy))), int(round(radius))


def crop_to_retina(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tight square crop around the retina FOV circle.

    Returns:
        (cropped_img, fov_mask): fov_mask is a boolean array matching
        cropped_img's first two dims, True inside the retina circle. By
        construction this mask is always the circle inscribed in the square
        crop -- see `cached_fov_mask`.
    """
    (cx, cy), radius = detect_retina_circle(img)
    x0, x1 = cx - radius, cx + radius
    y0, y1 = cy - radius, cy + radius

    h, w = img.shape[:2]
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - w), max(0, y1 - h)

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        x0, x1 = x0 + pad_left, x1 + pad_left
        y0, y1 = y0 + pad_top, y1 + pad_top

    cropped = img[y0:y1, x0:x1]
    mask = cached_fov_mask(cropped.shape[0])
    return cropped, mask


def cached_fov_mask(size: int) -> np.ndarray:
    """The FOV mask for a square retina crop of side `size`.

    `crop_to_retina` always yields a square crop whose FOV is exactly the
    inscribed circle, and resizing preserves that -- so this mask is a fixed
    function of `size` alone, independent of image content. Used to avoid
    re-detecting or re-storing the mask for cached images.
    """
    yy, xx = np.ogrid[:size, :size]
    center = size / 2
    return (xx - center) ** 2 + (yy - center) ** 2 <= center**2


def resize_cached(img: np.ndarray, _mask: np.ndarray, size: int = 448) -> tuple[np.ndarray, np.ndarray]:
    """Resize an image to (size, size) via Lanczos interpolation.

    `_mask` is expected to already be the exact inscribed circle for `img`'s
    current size (as produced by `crop_to_retina`). The resized mask is
    re-derived analytically at the target size via `cached_fov_mask` instead
    of interpolated pixel-by-pixel, since nearest-neighbor resizing an ideal
    circle only approximates that shape with boundary aliasing -- deriving
    it directly keeps every cached image's mask exactly reproducible from
    its size alone.
    """
    resized_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return resized_img, cached_fov_mask(size)


def preprocess_image(img: np.ndarray, size: int = 448) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the retina FOV and resize. Returns (uint8 RGB image, boolean FOV mask)."""
    cropped, mask = crop_to_retina(img)
    return resize_cached(cropped, mask, size=size)


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(42)
    t0 = time.time()
    for _ in range(5):
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.circle(canvas, (400, 300), 250, (120, 60, 40), -1)
        noise = rng.integers(0, 40, canvas.shape, dtype=np.uint8)
        canvas = cv2.add(canvas, noise)
        out_img, out_mask = preprocess_image(canvas)
        assert out_img.shape == (448, 448, 3)
        assert out_mask.shape == (448, 448)
        assert np.array_equal(out_mask, cached_fov_mask(448))
    elapsed = time.time() - t0
    print(f"retina.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
