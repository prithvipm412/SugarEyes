"""Cache the real Kaggle-trained grading model's held-out validation images
(data/splits/aptos_val.csv, downloaded via
`download_aptos_sample.py --filenames-csv`) into their own manifest.

Kept entirely separate from the local 300-image dev sample's cache/manifest
-- mixing them would risk evaluating the Kaggle-trained model against images
that were actually in its own training set, since the local dev sample was
drawn independently and may overlap the full dataset's training portion.
"""
from pathlib import Path

import cv2
import pandas as pd

from src.drscreen.preprocess.retina import preprocess_image

RAW_DIR = Path("data/raw/aptos_val")
CACHE_DIR = Path("data/cache/aptos_val")
MANIFEST_PATH = Path("data/cache/aptos_val_manifest.csv")


def main() -> None:
    labels = pd.read_csv(RAW_DIR / "labels.csv")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in labels.iterrows():
        img = cv2.imread(str(RAW_DIR / "images" / row["filename"]))
        if img is None:
            print(f"WARNING: unreadable image, skipping: {row['filename']}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cached_img, _mask = preprocess_image(img, size=448)
        out_path = CACHE_DIR / row["filename"]
        cv2.imwrite(str(out_path), cv2.cvtColor(cached_img, cv2.COLOR_RGB2BGR))
        rows.append({"path": str(out_path), "dataset": "aptos_val", "split": "val", "grade": row["grade"], "has_masks": False})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"cached {len(manifest)} images -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
