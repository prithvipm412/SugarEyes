"""Structured feature vector bridging Phase 3 (lesion detection) to Phase 4
(fusion). Fixed-length, real-valued, NaN-free -- computed from whatever
lesion masks, vessel mask, and structure locations the rest of lesions/
produces for a given image. A class with no detections contributes zeros,
never NaN.
"""
import numpy as np
from scipy import ndimage

from .vessels import vessel_metrics

LESION_CLASSES = ["ma", "he", "ex", "se"]

FEATURE_NAMES = (
    [f"{cls}_count" for cls in LESION_CLASSES]
    + [f"{cls}_total_area" for cls in LESION_CLASSES]
    + [f"{cls}_mean_dist_to_fovea" for cls in LESION_CLASSES]
    + [f"{cls}_max_size" for cls in LESION_CLASSES]
    + ["vessel_tortuosity", "vessel_branching_density", "nv_flag", "ma_count_macular"]
)


def _lesion_stats(mask: np.ndarray, fovea_center: tuple[float, float]) -> dict:
    labeled, n_labels = ndimage.label(mask.astype(bool))
    if n_labels == 0:
        return {"count": 0, "total_area": 0.0, "mean_dist_to_fovea": 0.0, "max_size": 0.0}

    fx, fy = fovea_center
    areas, dists = [], []
    for label_id in range(1, n_labels + 1):
        ys, xs = np.where(labeled == label_id)
        areas.append(len(xs))
        dists.append(float(np.hypot(xs.mean() - fx, ys.mean() - fy)))
    return {"count": n_labels, "total_area": float(sum(areas)), "mean_dist_to_fovea": float(np.mean(dists)), "max_size": float(max(areas))}


def extract_lesion_features(
    lesion_masks: dict[str, np.ndarray],
    vessel_mask: np.ndarray,
    fovea_center: tuple[float, float],
    od_center: tuple[float, float],
    od_radius: float,
    nv_result: dict,
) -> np.ndarray:
    """Fixed-length feature vector, in `FEATURE_NAMES` order."""
    per_class = {cls: _lesion_stats(lesion_masks.get(cls, np.zeros_like(vessel_mask)), fovea_center) for cls in LESION_CLASSES}

    values = (
        [per_class[cls]["count"] for cls in LESION_CLASSES]
        + [per_class[cls]["total_area"] for cls in LESION_CLASSES]
        + [per_class[cls]["mean_dist_to_fovea"] for cls in LESION_CLASSES]
        + [per_class[cls]["max_size"] for cls in LESION_CLASSES]
    )

    v_metrics = vessel_metrics(vessel_mask)
    values += [v_metrics["tortuosity"], v_metrics["branching_density"], float(nv_result["flag"])]

    # MA count in the macular region: MAs within 2 disc diameters of the
    # fovea, clinically the ones most tied to visual-acuity risk.
    ma_mask = lesion_masks.get("ma", np.zeros_like(vessel_mask)).astype(bool)
    labeled, n_labels = ndimage.label(ma_mask)
    macular_radius = 2 * (2 * od_radius)
    fx, fy = fovea_center
    macular_count = sum(
        1
        for label_id in range(1, n_labels + 1)
        for ys, xs in [np.where(labeled == label_id)]
        if np.hypot(xs.mean() - fx, ys.mean() - fy) <= macular_radius
    )
    values.append(float(macular_count))

    vec = np.array(values, dtype=np.float32)
    assert not np.isnan(vec).any(), "extract_lesion_features produced a NaN"
    return vec


if __name__ == "__main__":
    import time

    import cv2

    t0 = time.time()
    rng = np.random.default_rng(15)
    h, w = 300, 300

    for _ in range(20):
        lesion_masks = {}
        for cls in LESION_CLASSES:
            mask = np.zeros((h, w), dtype=np.uint8)
            if rng.random() < 0.7:  # some classes absent on some images, by design
                for _ in range(int(rng.integers(0, 5))):
                    cv2.circle(mask, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(2, 8)), 255, -1)
            lesion_masks[cls] = mask.astype(bool)

        vessel_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(vessel_mask, (50, 50), (250, 250), 1, 2)
        vessel_mask = vessel_mask.astype(bool)

        fovea_center = (float(rng.integers(0, w)), float(rng.integers(0, h)))
        od_center = (float(rng.integers(0, w)), float(rng.integers(0, h)))
        od_radius = float(rng.uniform(10, 30))
        nv_result = {"flag": bool(rng.random() < 0.5), "rule_based": True}

        vec = extract_lesion_features(lesion_masks, vessel_mask, fovea_center, od_center, od_radius, nv_result)
        assert vec.shape == (len(FEATURE_NAMES),), vec.shape
        assert not np.isnan(vec).any()

    # all-empty case (no lesions, no vessels at all) must still work cleanly
    empty_masks = {cls: np.zeros((h, w), dtype=bool) for cls in LESION_CLASSES}
    vec = extract_lesion_features(empty_masks, np.zeros((h, w), dtype=bool), (150.0, 150.0), (150.0, 150.0), 20.0, {"flag": False, "rule_based": True})
    assert vec.shape == (len(FEATURE_NAMES),)
    assert not np.isnan(vec).any()
    assert vec.sum() == 0.0

    elapsed = time.time() - t0
    print(f"features.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
