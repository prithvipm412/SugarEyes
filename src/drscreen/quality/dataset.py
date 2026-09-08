"""IQA dataset: synthetic GRADEABLE / ENHANCEABLE / REJECT labels derived from
camera-simulator severity bands.

Real clinical quality labels (EyeQ, over EyePACS) are Kaggle-only per
AGENTS.md and are not available for local training. This synthetic scheme is
what the Phase 1 local model and gate are trained and measured against -- it
must never be presented as clinical ground truth.
"""
import hashlib

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..preprocess.degrade import simulate_capture
from ..preprocess.retina import cached_fov_mask
from .features import extract_features

# label -> (severity_lo, severity_hi) for simulate_capture
SEVERITY_BANDS = {
    0: (0.0, 0.0),  # GRADEABLE
    1: (0.25, 0.45),  # ENHANCEABLE
    2: (0.70, 0.95),  # REJECT
}


def _deterministic_seed(path: str, label: int) -> int:
    digest = hashlib.sha256(f"{path}:{label}".encode()).hexdigest()
    return int(digest[:8], 16)


class IQADataset(Dataset):
    """Each cached image yields one sample per severity band (3x the manifest
    row count), so classes are exactly balanced by construction."""

    def __init__(self, manifest_csv: str, split: str, image_size: int = 224):
        manifest = pd.read_csv(manifest_csv)
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError(f"No rows for split={split!r} in {manifest_csv}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows) * len(SEVERITY_BANDS)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row_idx, label = divmod(idx, len(SEVERITY_BANDS))
        path = self.rows.loc[row_idx, "path"]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        lo, hi = SEVERITY_BANDS[label]
        seed = _deterministic_seed(path, label)
        severity = float(np.random.default_rng(seed).uniform(lo, hi)) if hi > lo else 0.0
        degraded = simulate_capture(img, severity=severity, seed=seed) if severity > 0 else img

        mask = cached_fov_mask(degraded.shape[0])
        features = extract_features(degraded, mask)

        resized = cv2.resize(degraded, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor, torch.from_numpy(features), label


if __name__ == "__main__":
    import tempfile
    import time
    from pathlib import Path

    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rows = []
        for i in range(5):
            canvas = np.zeros((448, 448, 3), dtype=np.uint8)
            cv2.circle(canvas, (224, 224), 220, (120, 60, 40), -1)
            noise = rng.integers(0, 60, canvas.shape, dtype=np.uint8)
            canvas = cv2.add(canvas, noise)
            img_path = tmp_path / f"img_{i}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            rows.append({"path": str(img_path), "dataset": "synthetic", "split": "train", "grade": 0, "has_masks": False})
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

        ds = IQADataset(str(manifest_path), split="train")
        assert len(ds) == 5 * len(SEVERITY_BANDS)
        t0 = time.time()
        for i in range(5):
            tensor, features, label = ds[i]
            assert tensor.shape == (3, 224, 224)
            assert not torch.isnan(features).any()
            assert label in (0, 1, 2)
        elapsed = time.time() - t0
    print(f"dataset.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
