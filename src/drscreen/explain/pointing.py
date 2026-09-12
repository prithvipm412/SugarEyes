"""Pointing-game CAM/lesion agreement metric.

A 7x7 or 14x14 Grad-CAM upsampled to full resolution scores near zero
against 15-pixel microaneurysms under IoU even when the model is behaving
reasonably (see AGENTS.md) -- IoU compares two *areas*, and the CAM's area
is defined mostly by upsampling artifacts at that scale. Pointing-game
instead asks whether each lesion's *centroid* falls inside the CAM's
most-activated region -- meaningful regardless of how coarse that region's
boundary is.
"""
import cv2
import numpy as np
from scipy import ndimage


def pointing_game(cam: np.ndarray, lesion_mask: np.ndarray, top_k: float = 0.10) -> dict:
    """Fraction of connected lesion components whose centroid falls inside
    the top `top_k` fraction of CAM activation (resized to `lesion_mask`'s
    shape if they differ).

    Returns:
        {"hits": int, "total": int, "score": float} -- score is NaN when
        there are no lesion components in this mask (undefined, not zero:
        "no lesions, so no evidence for or against the model's attention").
    """
    if cam.shape != lesion_mask.shape:
        cam = cv2.resize(cam.astype(np.float32), (lesion_mask.shape[1], lesion_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    threshold = np.quantile(cam, 1 - top_k)
    top_region = cam >= threshold

    labeled, n_labels = ndimage.label(lesion_mask.astype(bool))
    if n_labels == 0:
        return {"hits": 0, "total": 0, "score": float("nan")}

    hits = 0
    for label_id, sl in enumerate(ndimage.find_objects(labeled, max_label=n_labels), start=1):
        if sl is None:
            continue
        region = labeled[sl] == label_id
        ys, xs = np.where(region)
        cy = sl[0].start + int(round(ys.mean()))
        cx = sl[1].start + int(round(xs.mean()))
        if top_region[cy, cx]:
            hits += 1
    return {"hits": hits, "total": n_labels, "score": hits / n_labels}


def pointing_game_per_class(cam: np.ndarray, lesion_masks: dict[str, np.ndarray], top_k: float = 0.10) -> dict:
    """`pointing_game` for every lesion class in `lesion_masks`. Every class
    passed in gets a result -- none silently skipped, even when its score is
    expected to be poor (microaneurysms, per AGENTS.md -- that gap between
    exudate/haemorrhage scores and microaneurysm scores IS the finding, not
    a bug to hide)."""
    return {cls: pointing_game(cam, mask, top_k=top_k) for cls, mask in lesion_masks.items()}


if __name__ == "__main__":
    import time

    t0 = time.time()
    rng = np.random.default_rng(20)

    # A CAM strongly activated in the top-left quadrant.
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[0:30, 0:30] = 1.0
    cam += rng.uniform(0, 0.05, cam.shape)  # a little noise, not a perfect step

    # Lesion inside the hot region -> should score 1.0
    mask_inside = np.zeros((100, 100), dtype=np.uint8)
    mask_inside[10:15, 10:15] = 1
    result = pointing_game(cam, mask_inside.astype(bool), top_k=0.10)
    assert result["total"] == 1 and result["score"] == 1.0, result

    # Lesion far outside the hot region -> should score 0.0
    mask_outside = np.zeros((100, 100), dtype=np.uint8)
    mask_outside[80:85, 80:85] = 1
    result = pointing_game(cam, mask_outside.astype(bool), top_k=0.10)
    assert result["total"] == 1 and result["score"] == 0.0, result

    # No lesions at all -> NaN, not zero
    empty_mask = np.zeros((100, 100), dtype=bool)
    result = pointing_game(cam, empty_mask, top_k=0.10)
    assert result["total"] == 0 and np.isnan(result["score"])

    # Multiple classes, one with no lesions -- none silently dropped
    per_class = pointing_game_per_class(cam, {"ma": mask_inside.astype(bool), "he": empty_mask, "ex": mask_outside.astype(bool)})
    assert set(per_class.keys()) == {"ma", "he", "ex"}
    assert per_class["ma"]["score"] == 1.0
    assert np.isnan(per_class["he"]["score"])
    assert per_class["ex"]["score"] == 0.0

    # cam/mask shape mismatch is resized, not an error
    small_cam = cv2.resize(cam, (25, 25))
    result = pointing_game(small_cam, mask_inside.astype(bool), top_k=0.10)
    assert result["total"] == 1

    elapsed = time.time() - t0
    print(f"pointing.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
