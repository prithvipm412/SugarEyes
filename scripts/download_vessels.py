"""Download DRIVE + CHASE_DB1 + STARE vessel ground truth (~88 images total,
see AGENTS.md) from their Kaggle mirrors and normalize into a common layout:

  data/raw/vessels/<dataset>/images/<id>.png
  data/raw/vessels/<dataset>/masks/<id>.png   (binary vessel mask, 0/255)

<dataset> in {drive, chasedb1, stare}. DRIVE's official test set (20 images)
has no publicly released ground truth (held out for the original online
challenge) -- only its 20 training images (which do have ground truth) are
included here.

Requires Kaggle API credentials (see download_aptos_sample.py).
"""
import shutil
from pathlib import Path

import cv2
import numpy as np

RAW_DIR = Path("data/raw/vessels")

DATASETS = {
    "drive": "andrewmvd/drive-digital-retinal-images-for-vessel-extraction",
    "chasedb1": "buffyhridoy/chase-db1",
    "stare": "vasavigneswar/stare-retinal-dataset",
}


def _binarize_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"could not read mask: {mask_path}")
    return (mask > 127).astype(np.uint8) * 255


def _write_pair(out_dir: Path, image_id: str, image: np.ndarray, mask: np.ndarray) -> None:
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "images" / f"{image_id}.png"), image)
    cv2.imwrite(str(out_dir / "masks" / f"{image_id}.png"), mask)


def _download(dataset_slug: str, dest: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {dataset_slug}...")
    api.dataset_download_files(dataset_slug, path=str(dest), unzip=True)


def process_drive(download_dir: Path, out_dir: Path) -> int:
    train_dir = next(download_dir.rglob("training/images")).parent
    images_dir = train_dir / "images"
    manual_dir = train_dir / "1st_manual"
    count = 0
    for img_path in sorted(images_dir.glob("*_training.tif")):
        image_id = img_path.stem.replace("_training", "")
        manual_path = manual_dir / f"{image_id}_manual1.gif"
        image = cv2.imread(str(img_path))
        mask = _binarize_mask(manual_path)
        _write_pair(out_dir, f"drive_{image_id}", image, mask)
        count += 1
    return count


def process_chasedb1(download_dir: Path, out_dir: Path) -> int:
    images_dir = next(download_dir.rglob("Images"))
    masks_dir = next(download_dir.rglob("Masks"))
    count = 0
    for img_path in sorted(images_dir.glob("*.jpg")):
        image_id = img_path.stem
        mask_path = masks_dir / f"{image_id}_1stHO.png"
        image = cv2.imread(str(img_path))
        mask = _binarize_mask(mask_path)
        _write_pair(out_dir, f"chasedb1_{image_id}", image, mask)
        count += 1
    return count


def process_stare(download_dir: Path, out_dir: Path) -> int:
    images_dir = next(download_dir.rglob("stare-images"))
    labels_dir = next(download_dir.rglob("labels-ah"))
    count = 0
    for mask_path in sorted(labels_dir.glob("*.ah.ppm")):
        image_id = mask_path.stem.replace(".ah", "")
        img_path = images_dir / f"{image_id}.ppm"
        image = cv2.imread(str(img_path))
        mask = _binarize_mask(mask_path)
        _write_pair(out_dir, f"stare_{image_id}", image, mask)
        count += 1
    return count


PROCESSORS = {"drive": process_drive, "chasedb1": process_chasedb1, "stare": process_stare}


def main() -> None:
    for name, slug in DATASETS.items():
        download_dir = RAW_DIR / f"_download_{name}"
        _download(slug, download_dir)
        out_dir = RAW_DIR / name
        n = PROCESSORS[name](download_dir, out_dir)
        shutil.rmtree(download_dir)
        print(f"{name}: wrote {n} image/mask pairs to {out_dir}")


if __name__ == "__main__":
    main()
