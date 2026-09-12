"""Fit the official referable-DR calibration temperature on the full real
APTOS validation set (the same 550-image set used for threshold selection
in Phase 2), and freeze it to models/manifest.json alongside the threshold.
Also writes before/after reliability diagrams as real artifacts.
"""
import json
from pathlib import Path

import numpy as np

from src.drscreen.explain.calibrate import apply_temperature, expected_calibration_error, fit_temperature, reliability_diagram

FUSION_DATA = Path("data/cache/fusion_dataset.npz")
MANIFEST_PATH = Path("models/manifest.json")
OUT_DIR = Path("models/calibration")


def main() -> None:
    data = np.load(FUSION_DATA)
    labels = (data["grades"] >= 2).astype(int)
    raw_scores = data["referable_scores"]

    ece_before = expected_calibration_error(raw_scores, labels)
    temperature = fit_temperature(raw_scores, labels)
    calibrated = apply_temperature(raw_scores, temperature)
    ece_after = expected_calibration_error(calibrated, labels)

    print(f"temperature={temperature:.4f}  ECE before={ece_before:.4f}  ECE after={ece_after:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reliability_diagram(raw_scores, labels, str(OUT_DIR / "reliability_before.png"), title="Reliability (uncalibrated)")
    reliability_diagram(calibrated, labels, str(OUT_DIR / "reliability_after.png"), title="Reliability (temperature-scaled)")

    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    manifest["calibration_temperature"] = temperature
    manifest["calibration_ece_before"] = ece_before
    manifest["calibration_ece_after"] = ece_after
    manifest["calibration_n_images"] = int(len(labels))
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"wrote calibration_temperature={temperature:.4f} to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
