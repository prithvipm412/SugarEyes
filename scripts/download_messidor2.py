"""Download Messidor-2 and normalize into data/raw/messidor2/.

The images and labels ship as two separate Kaggle mirrors because the
canonical, most-cited label source for Messidor-2 is not from ADCIS's own
distribution but from Google Brain's re-adjudication of it:

  Krause, J. et al. Grader variability and the importance of reference
  standards for evaluating machine learning models for diabetic
  retinopathy. Ophthalmology (2018). doi:10.1016/j.ophtha.2018.01.034

This project could not complete ADCIS's own registration process in its
build environment (see data/splits/README.md), so this mirror is the
substitute -- chosen deliberately for using Google Brain's adjudicated
grades rather than the original, noisier Messidor-2 grades: three
ophthalmologists graded each image independently with disagreements
adjudicated by a retinal specialist, which is why this is the standard
label source for Messidor-2 throughout the DR-screening ML literature, not
a lower-quality stand-in.

  data/raw/messidor2/images/<filename>.png   (raw, original resolution --
      NOT any Kaggle mirror's own preprocessed version, so this project's
      own preprocess_image crop+resize is what actually runs on it, same
      as every other dataset)
  data/raw/messidor2/labels.csv              (filename, grade) -- gradable
      images only (adjudicated_gradable == 1); ungradable images have no
      grade in the source data at all, so there is nothing to include them
      as, the same way this project's own IQA quality gate would route
      them to REJECT rather than grade them.

Requires Kaggle API credentials (see download_aptos_sample.py).
"""
import shutil
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/messidor2")
IMAGES_SLUG = "xyaustin/messidor2"  # raw/original-resolution images
GRADES_SLUG = "google-brain/messidor2-dr-grades"  # Krause et al. 2018 adjudicated grades


def process_grades(grades_dir: Path) -> pd.DataFrame:
    grades = pd.read_csv(grades_dir / "messidor_data.csv")
    gradable = grades[grades["adjudicated_gradable"] == 1].copy()
    labels = gradable.rename(columns={"image_id": "filename", "adjudicated_dr_grade": "grade"})[["filename", "grade"]]
    labels["grade"] = labels["grade"].astype(int)
    return labels.reset_index(drop=True)


def process_images(images_download_dir: Path, labels: pd.DataFrame, out_dir: Path) -> int:
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    src_root = next(images_download_dir.rglob("images"))  # messidor-2/images/<file>.png in the zip
    n_copied = 0
    for filename in labels["filename"]:
        src = src_root / filename
        if not src.exists():
            print(f"WARNING: {filename} in grades CSV but not in image mirror, skipping")
            continue
        shutil.copy(src, out_images / filename)
        n_copied += 1
    return n_copied


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    download_dir = RAW_DIR / "_download"
    grades_dir = download_dir / "grades"
    images_dir = download_dir / "images"
    grades_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading {GRADES_SLUG}...")
    api.dataset_download_files(GRADES_SLUG, path=str(grades_dir), unzip=True)
    labels = process_grades(grades_dir)

    print(f"downloading {IMAGES_SLUG} (~2.3GB, raw resolution)...")
    api.dataset_download_files(IMAGES_SLUG, path=str(images_dir), unzip=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    n_copied = process_images(images_dir, labels, RAW_DIR)
    labels = labels[labels["filename"].isin({p.name for p in (RAW_DIR / "images").glob("*.png")})].reset_index(drop=True)
    labels.to_csv(RAW_DIR / "labels.csv", index=False)

    print(f"wrote {n_copied} images and {len(labels)} labels to {RAW_DIR}")
    print(f"grade distribution:\n{labels['grade'].value_counts().sort_index()}")

    shutil.rmtree(download_dir)


if __name__ == "__main__":
    main()
