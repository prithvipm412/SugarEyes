"""DR grading dataset: cached retina images + ICDR grade labels, with
albumentations augmentation plus low-severity synthetic camera degradation."""
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..preprocess.degrade import simulate_capture


def build_transform(image_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.Rotate(limit=30, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomScale(scale_limit=0.1, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
                A.Resize(image_size, image_size),
            ]
        )
    return A.Compose([A.Resize(image_size, image_size)])


class GradingDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        split: str,
        image_size: int = 224,
        train: bool = False,
        degrade_prob: float = 0.3,
        degrade_severity: float = 0.3,
        seed: int = 42,
    ):
        manifest = pd.read_csv(manifest_csv)
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError(f"No rows for split={split!r} in {manifest_csv}")
        self.transform = build_transform(image_size, train)
        self.train = train
        self.degrade_prob = degrade_prob
        self.degrade_severity = degrade_severity
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows.loc[idx]
        img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)

        if self.train and self.rng.random() < self.degrade_prob:
            severity = float(self.rng.uniform(0.05, self.degrade_severity))
            img = simulate_capture(img, severity=severity, seed=int(self.rng.integers(0, 2**31)))

        img = self.transform(image=img)["image"]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor, int(row["grade"])


def class_weights(rows: pd.DataFrame, num_grades: int = 5) -> np.ndarray:
    """Per-sample weight = inverse frequency of that sample's grade, for
    WeightedRandomSampler -- APTOS is ~50% grade 0, and a naive model trained
    without this collapses onto an all-zero predictor (see AGENTS.md)."""
    counts = rows["grade"].value_counts().reindex(range(num_grades), fill_value=0)
    freq = counts / counts.sum()
    weight_per_grade = (1.0 / freq.replace(0, np.inf)).to_dict()
    return np.array(rows["grade"].map(weight_per_grade), dtype=np.float64)


if __name__ == "__main__":
    import tempfile
    import time
    from pathlib import Path

    rng = np.random.default_rng(6)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rows = []
        for i in range(5):
            canvas = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.circle(canvas, (150, 150), 140, (110, 50, 35), -1)
            canvas = cv2.add(canvas, rng.integers(0, 40, canvas.shape, dtype=np.uint8))
            img_path = tmp_path / f"img_{i}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            rows.append({"path": str(img_path), "dataset": "synthetic", "split": "train", "grade": i % 5, "has_masks": False})
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

        ds = GradingDataset(str(manifest_path), split="train", image_size=224, train=True)
        assert len(ds) == 5
        weights = class_weights(ds.rows)
        assert weights.shape == (5,)
        assert np.isfinite(weights).all()

        t0 = time.time()
        for i in range(5):
            tensor, grade = ds[i]
            assert tensor.shape == (3, 224, 224)
            assert grade in range(5)
        elapsed = time.time() - t0
    print(f"dataset.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
