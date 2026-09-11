"""Optic disc and fovea localization.

Disc: brightest-blob candidate scoring combined with vessel convergence
density (vessels anatomically converge at the disc). Fovea: darkest,
vessel-sparse region searched ~2.5 disc diameters either side of the disc
along the horizontal meridian (image laterality -- which side is temporal --
isn't reliably known from the image alone, so both sides are tried).

Both degrade gracefully: a fundus photo with no clear disc still returns a
best-effort location with a low confidence score, never an exception -- a
poor-quality image is an expected input, not a programming error.
"""
import cv2
import numpy as np
from scipy import ndimage


def _bright_blob_mask(gray: np.ndarray, min_area: int = 20, max_area_frac: float = 0.15) -> np.ndarray:
    """Threshold `gray` to isolate bright blobs, searching progressively
    looser percentiles until at least one blob of plausible disc size
    appears. A single fixed percentile (e.g. 98th) is fragile: a real optic
    disc typically covers ~2% of a well-cropped frame, so a fixed threshold
    can land exactly on the background level and select the whole image."""
    total_pixels = gray.size
    for percentile in (99.9, 99.7, 99.5, 99.0, 98.0, 95.0, 90.0):
        threshold = np.percentile(gray, percentile)
        bright_mask = gray >= max(threshold, 1)
        labeled, n_labels = ndimage.label(bright_mask)
        if n_labels == 0:
            continue
        sizes = ndimage.sum(bright_mask, labeled, range(1, n_labels + 1))
        sizes = np.asarray(sizes)
        if ((sizes >= min_area) & (sizes <= max_area_frac * total_pixels)).any():
            return bright_mask
    # Last resort: the loosest threshold, whatever it gives.
    threshold = np.percentile(gray, 90)
    return gray >= max(threshold, 1)


def locate_optic_disc(img: np.ndarray, vessel_map: np.ndarray) -> dict:
    """Locate the optic disc.

    Args:
        img: RGB uint8 fundus image, shape (H, W, 3).
        vessel_map: boolean/0-1 vessel segmentation, shape (H, W).

    Returns:
        {"center": (x, y), "radius": int, "confidence": float in [0, 1]}
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    vessel_map = vessel_map.astype(bool)

    bright_mask = _bright_blob_mask(gray)
    labeled, n_labels = ndimage.label(bright_mask)
    max_area = 0.15 * gray.size

    best_score, best = -np.inf, None
    for label_id in range(1, n_labels + 1):
        ys, xs = np.where(labeled == label_id)
        area = len(xs)
        if area < 10 or area > max_area:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        radius = float(np.sqrt(area / np.pi))

        bbox_w, bbox_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        circularity = area / (bbox_w * bbox_h)

        neighborhood_radius = max(radius * 2, 10)
        yy, xx = np.ogrid[:h, :w]
        neighborhood = (xx - cx) ** 2 + (yy - cy) ** 2 <= neighborhood_radius**2
        vessel_density = vessel_map[neighborhood].mean() if neighborhood.any() else 0.0

        brightness = gray[ys, xs].mean() / 255.0
        score = 0.4 * brightness + 0.3 * circularity + 0.3 * vessel_density
        if score > best_score:
            best_score = score
            best = {"center": (cx, cy), "radius": radius}

    if best is None:
        cy, cx = np.unravel_index(np.argmax(gray), gray.shape)
        return {"center": (int(cx), int(cy)), "radius": int(0.05 * min(h, w)), "confidence": 0.0}

    return {
        "center": (int(round(best["center"][0])), int(round(best["center"][1]))),
        "radius": int(round(best["radius"])),
        "confidence": float(np.clip(best_score, 0.0, 1.0)),
    }


def locate_fovea(img: np.ndarray, od_center: tuple[int, int], od_radius: int, vessel_map: np.ndarray) -> dict:
    """Locate the fovea relative to an already-located optic disc.

    Args:
        img: RGB uint8 fundus image.
        od_center: (x, y) optic disc center, from `locate_optic_disc`.
        od_radius: optic disc radius, from `locate_optic_disc`.
        vessel_map: boolean/0-1 vessel segmentation, shape (H, W).

    Returns:
        {"center": (x, y), "confidence": float in [0, 1]}
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    od_x, od_y = od_center
    search_offset = 2.5 * (2 * od_radius)
    window = max(od_radius, 10)
    vessel_map = vessel_map.astype(bool)

    best_score, best_center = -np.inf, None
    for direction in (-1, 1):
        cx, cy = od_x + direction * search_offset, od_y
        if not (0 <= cx < w and 0 <= cy < h):
            continue

        y0, y1 = int(max(0, cy - window)), int(min(h, cy + window))
        x0, x1 = int(max(0, cx - window)), int(min(w, cx + window))
        if y1 <= y0 or x1 <= x0:
            continue

        window_gray = gray[y0:y1, x0:x1]
        window_vessel = vessel_map[y0:y1, x0:x1]
        darkness = 1.0 - window_gray.mean() / 255.0
        avascularity = 1.0 - window_vessel.mean()
        score = 0.6 * darkness + 0.4 * avascularity
        if score > best_score:
            best_score = score
            best_center = (x0 + window_gray.shape[1] / 2, y0 + window_gray.shape[0] / 2)

    if best_center is None:
        return {"center": (int(od_x), int(od_y)), "confidence": 0.0}

    return {
        "center": (int(round(best_center[0])), int(round(best_center[1]))),
        "confidence": float(np.clip(best_score, 0.0, 1.0)),
    }


if __name__ == "__main__":
    import time

    t0 = time.time()
    for _ in range(5):
        h, w = 448, 448
        img = np.full((h, w, 3), 40, dtype=np.uint8)  # dim retina background
        true_od = (330, 224)
        cv2.circle(img, true_od, 35, (230, 210, 120), -1)  # bright disc

        vessel_canvas = np.zeros((h, w), dtype=np.uint8)
        rng = np.random.default_rng(0)
        for _ in range(12):
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(60, 150)
            end = (int(true_od[0] + length * np.cos(angle)), int(true_od[1] + length * np.sin(angle)))
            cv2.line(vessel_canvas, true_od, end, 1, 2)
        vessel_map = vessel_canvas.astype(bool)

        true_fovea_region = (true_od[0] - 200, true_od[1])
        cv2.circle(img, true_fovea_region, 25, (15, 10, 8), -1)  # darker, avascular

        od_result = locate_optic_disc(img, vessel_map)
        assert abs(od_result["center"][0] - true_od[0]) < 20, od_result
        assert abs(od_result["center"][1] - true_od[1]) < 20, od_result
        assert 0.0 <= od_result["confidence"] <= 1.0

        fovea_result = locate_fovea(img, od_result["center"], od_result["radius"], vessel_map)
        assert fovea_result["center"][0] < od_result["center"][0], fovea_result  # found on the darker (left) side
        assert 0.0 <= fovea_result["confidence"] <= 1.0

        # Graceful degradation: an all-uniform image has no clear disc, but
        # must still return a result (low confidence), not raise.
        blank = np.full((h, w, 3), 50, dtype=np.uint8)
        blank_result = locate_optic_disc(blank, np.zeros((h, w), dtype=bool))
        assert blank_result["confidence"] <= 0.5

    elapsed = time.time() - t0
    print(f"structures.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
