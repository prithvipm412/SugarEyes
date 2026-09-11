"""Download IDRiD (516 graded, 81 with pixel masks for MA/HE/EX/SE/OD, plus
OD + fovea center ground truth for 516 images) from its Kaggle mirror and
normalize into:

  data/raw/idrid/segmentation/images/<id>.jpg
  data/raw/idrid/segmentation/masks/{ma,he,ex,se,od}/<id>.png   (binary, 0/255;
      not every image has every class -- soft exudates in particular are rare,
      see AGENTS.md)
  data/raw/idrid/segmentation/split.csv          (id, split) -- IDRiD's own
      official train(54)/test(27) split, kept as train/val here
  data/raw/idrid/localization/images/<id>.jpg
  data/raw/idrid/localization/labels.csv         (id, split, od_x, od_y, fovea_x, fovea_y)

Requires Kaggle API credentials (see download_aptos_sample.py).
"""
import shutil
from pathlib import Path

import cv2
import pandas as pd

RAW_DIR = Path("data/raw/idrid")
DATASET_SLUG = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"

SEG_CLASSES = {"ma": "Microaneurysms", "he": "Haemorrhages", "ex": "Hard Exudates", "se": "Soft Exudates", "od": "Optic Disc"}
SEG_SUFFIX = {"ma": "MA", "he": "HE", "ex": "EX", "se": "SE", "od": "OD"}


def _binarize_mask(mask_path: Path) -> "cv2.typing.MatLike":
    import numpy as np

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return (mask > 0).astype(np.uint8) * 255


def process_segmentation(root: Path, out_dir: Path) -> pd.DataFrame:
    seg_root = next(root.rglob("A. Segmentation"))
    rows = []
    for split_name, set_name in [("train", "a. Training Set"), ("val", "b. Testing Set")]:
        images_dir = seg_root / "1. Original Images" / set_name
        for img_path in sorted(images_dir.glob("*.jpg")):
            image_id = img_path.stem
            out_images = out_dir / "images"
            out_images.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, out_images / f"{image_id}.jpg")

            has_mask = {}
            for cls, folder_name in SEG_CLASSES.items():
                folder_number = list(SEG_CLASSES).index(cls) + 1
                mask_dir = seg_root / "2. All Segmentation Groundtruths" / set_name / f"{folder_number}. {folder_name}"
                mask_path = mask_dir / f"{image_id}_{SEG_SUFFIX[cls]}.tif"
                out_mask_dir = out_dir / "masks" / cls
                out_mask_dir.mkdir(parents=True, exist_ok=True)
                if mask_path.exists():
                    cv2.imwrite(str(out_mask_dir / f"{image_id}.png"), _binarize_mask(mask_path))
                    has_mask[cls] = True
                else:
                    has_mask[cls] = False
            rows.append({"id": image_id, "split": split_name, **{f"has_{c}": has_mask[c] for c in SEG_CLASSES}})

    split_df = pd.DataFrame(rows)
    split_df.to_csv(out_dir / "split.csv", index=False)
    return split_df


def process_localization(root: Path, out_dir: Path) -> pd.DataFrame:
    """IDRiD's train and test localization CSVs both restart IDs from
    IDRiD_001 -- writing every image into one shared images/ folder would
    let the 103 test images silently overwrite the first 103 train images
    (same filenames), corrupting that many train label/image pairs. Images
    are kept in split-specific subdirectories to make that impossible.
    """
    loc_root = next(root.rglob("C. Localization"))

    rows = []
    for split_name, set_name, od_csv, fovea_csv in [
        ("train", "a. Training Set", "a. IDRiD_OD_Center_Training Set_Markups.csv", "IDRiD_Fovea_Center_Training Set_Markups.csv"),
        ("val", "b. Testing Set", "b. IDRiD_OD_Center_Testing Set_Markups.csv", "IDRiD_Fovea_Center_Testing Set_Markups.csv"),
    ]:
        images_dir = loc_root / "1. Original Images" / set_name
        od_df = pd.read_csv(loc_root / "2. Groundtruths" / "1. Optic Disc Center Location" / od_csv, usecols=[0, 1, 2])
        od_df.columns = ["id", "od_x", "od_y"]
        od_df = od_df.dropna(subset=["id"])
        fovea_df = pd.read_csv(loc_root / "2. Groundtruths" / "2. Fovea Center Location" / fovea_csv, usecols=[0, 1, 2])
        fovea_df.columns = ["id", "fovea_x", "fovea_y"]
        fovea_df = fovea_df.dropna(subset=["id"])
        merged = od_df.merge(fovea_df, on="id", how="inner")
        merged["split"] = split_name

        out_images = out_dir / "images" / split_name
        out_images.mkdir(parents=True, exist_ok=True)
        for _, row in merged.iterrows():
            src = images_dir / f"{row['id']}.jpg"
            if src.exists():
                shutil.copy(src, out_images / f"{row['id']}.jpg")
                rows.append(row.to_dict())

    labels_df = pd.DataFrame(rows)[["id", "split", "od_x", "od_y", "fovea_x", "fovea_y"]]
    labels_df.to_csv(out_dir / "labels.csv", index=False)
    return labels_df


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    download_dir = RAW_DIR / "_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {DATASET_SLUG}...")
    api.dataset_download_files(DATASET_SLUG, path=str(download_dir), unzip=True)

    seg_df = process_segmentation(download_dir, RAW_DIR / "segmentation")
    print(f"segmentation: wrote {len(seg_df)} images to {RAW_DIR / 'segmentation'}")

    loc_df = process_localization(download_dir, RAW_DIR / "localization")
    print(f"localization: wrote {len(loc_df)} images to {RAW_DIR / 'localization'}")

    shutil.rmtree(download_dir)


if __name__ == "__main__":
    main()
