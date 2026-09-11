"""Neovascularisation flag: rule-based only. No public pixel ground truth
exists for neovascularisation (see AGENTS.md), so this is abnormal vessel
tortuosity + branching density within one disc diameter of the optic disc,
thresholded -- never a learned model. Every output carries "rule_based":
True, and the UI must render it differently from learned outputs rather
than implying it came from a trained model.
"""
import numpy as np

from .vessels import vessel_metrics

# Placeholder defaults until calibrate_thresholds() is run for real against
# DRIVE/CHASE_DB1/STARE ground truth (all disease-free of proliferative NV).
DEFAULT_TORTUOSITY_THRESHOLD = 1.15
DEFAULT_BRANCHING_DENSITY_THRESHOLD = 0.08


def detect_neovascularisation(
    vessel_mask: np.ndarray,
    od_center: tuple[int, int],
    od_radius: int,
    tortuosity_threshold: float = DEFAULT_TORTUOSITY_THRESHOLD,
    branching_density_threshold: float = DEFAULT_BRANCHING_DENSITY_THRESHOLD,
) -> dict:
    """Flag abnormal vessel tortuosity/branching within one disc diameter of
    the optic disc -- a proxy for neovascularisation at the disc (NVD).

    Returns:
        {"flag": bool, "tortuosity": float, "branching_density": float, "rule_based": True}
    """
    h, w = vessel_mask.shape
    yy, xx = np.ogrid[:h, :w]
    od_x, od_y = od_center
    disc_diameter = 2 * od_radius
    neighborhood = (xx - od_x) ** 2 + (yy - od_y) ** 2 <= disc_diameter**2

    local_mask = np.zeros_like(vessel_mask, dtype=bool)
    local_mask[neighborhood] = vessel_mask.astype(bool)[neighborhood]

    metrics = vessel_metrics(local_mask)
    flag = bool(metrics["tortuosity"] > tortuosity_threshold or metrics["branching_density"] > branching_density_threshold)

    return {"flag": flag, "tortuosity": metrics["tortuosity"], "branching_density": metrics["branching_density"], "rule_based": True}


def calibrate_thresholds(normal_vessel_masks: list[np.ndarray], percentile: float = 95.0) -> dict:
    """Calibrate thresholds from KNOWN NORMAL (non-neovascular) vessel masks
    -- e.g. DRIVE/CHASE_DB1/STARE ground truth, none of which show
    proliferative disease. Returns the given percentile of each metric's
    distribution across those masks, to flag values above what's normally seen."""
    tortuosities, densities = [], []
    for mask in normal_vessel_masks:
        m = vessel_metrics(mask)
        if m["tortuosity"] > 0:
            tortuosities.append(m["tortuosity"])
        densities.append(m["branching_density"])
    return {
        "tortuosity_threshold": float(np.percentile(tortuosities, percentile)) if tortuosities else DEFAULT_TORTUOSITY_THRESHOLD,
        "branching_density_threshold": float(np.percentile(densities, percentile)) if densities else DEFAULT_BRANCHING_DENSITY_THRESHOLD,
    }


if __name__ == "__main__":
    import time

    import cv2

    t0 = time.time()
    od_center, od_radius = (200, 200), 30

    # Normal case: a few simple, non-tortuous vessels near the disc -> no flag
    normal_mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.line(normal_mask, od_center, (250, 180), 1, 2)
    cv2.line(normal_mask, od_center, (150, 220), 1, 2)
    result = detect_neovascularisation(normal_mask.astype(bool), od_center, od_radius)
    assert result["rule_based"] is True
    assert result["flag"] is False, result

    # Abnormal case: a dense tangle of short, sharply-angled segments near
    # the disc -> high branching density, should flag.
    rng = np.random.default_rng(14)
    tangle_mask = np.zeros((400, 400), dtype=np.uint8)
    for _ in range(40):
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(5, 15)
        end = (int(od_center[0] + length * np.cos(angle)), int(od_center[1] + length * np.sin(angle)))
        cv2.line(tangle_mask, od_center, end, 1, 1)
    result = detect_neovascularisation(tangle_mask.astype(bool), od_center, od_radius, branching_density_threshold=0.02)
    assert result["rule_based"] is True
    assert result["flag"] is True, result

    # calibrate_thresholds runs and returns finite values on a set of masks
    calibrated = calibrate_thresholds([normal_mask.astype(bool), tangle_mask.astype(bool)])
    assert np.isfinite(calibrated["tortuosity_threshold"])
    assert np.isfinite(calibrated["branching_density_threshold"])

    elapsed = time.time() - t0
    print(f"neovascularisation.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
