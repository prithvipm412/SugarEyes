"""Build the fusion/calibration/ablation dataset: for every real APTOS
validation image (the Kaggle-trained grading model's own held-out split,
see scripts/cache_aptos_val.py), run the full Phase 1-3 pipeline to get the
grading logits and lesion feature vector, plus the real ICDR grade label.

Writes data/cache/fusion_dataset.npz: grading_logits (N,4), lesion_features
(N,F), grades (N,), referable_scores (N,), image_ids (N,).
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.drscreen.pipeline import ScreeningPipeline

MANIFEST = Path("data/cache/aptos_val_manifest.csv")
OUT_PATH = Path("data/cache/fusion_dataset.npz")


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    pipeline = ScreeningPipeline()

    grading_logits, lesion_features, grades, referable_scores, image_ids = [], [], [], [], []
    t0 = time.time()
    for i, row in manifest.iterrows():
        img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
        logits, features, raw_score, _grade = pipeline.grade_and_extract_features(img)
        grading_logits.append(logits)
        lesion_features.append(features)
        grades.append(int(row["grade"]))
        referable_scores.append(raw_score)
        image_ids.append(Path(row["path"]).stem)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"{i + 1}/{len(manifest)}  ({elapsed / (i + 1):.2f}s/image)", flush=True)

    np.savez(
        OUT_PATH,
        grading_logits=np.stack(grading_logits),
        lesion_features=np.stack(lesion_features),
        grades=np.array(grades),
        referable_scores=np.array(referable_scores),
        image_ids=np.array(image_ids),
    )
    print(f"wrote {len(grades)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
