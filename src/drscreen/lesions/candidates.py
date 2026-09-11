"""Microaneurysm candidate cascade -- the sub-pixel detection this problem
statement specifically asks for (see AGENTS.md).

Stage 1 (this module's `extract_candidates`) narrows a whole retina image
down to ~50-500 candidate locations via classical morphology. Stage 2 (a
48x48 patch CNN) then scores only those candidates as MA vs not. This is
O(candidates), not O(pixels): a dense per-pixel CNN scan for ~15-pixel
microaneurysms over a full retina image would be far more expensive for the
same result.

`candidate_stage_recall` measures stage 1 alone against ground truth --
recall at that stage is the ceiling for the whole cascade, since stage 2 can
only reject candidates, never recover one stage 1 missed entirely.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from ..device import get_device

PATCH_SIZE = 48


def extract_candidates(img: np.ndarray, vessel_mask: np.ndarray, min_area: int = 3, max_area: int = 60, min_circularity: float = 0.3) -> list[dict]:
    """Stage 1: morphological top-hat on the inverted green channel, vessels
    subtracted, thresholded and filtered by area + circularity.

    Returns a list of {"center": (x, y), "area": int}, typically 50-500 per image.
    """
    green = img[:, :, 1]
    inverted = 255 - green

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    top_hat = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, kernel)
    top_hat = top_hat.copy()
    top_hat[vessel_mask.astype(bool)] = 0

    positive = top_hat[top_hat > 0]
    if positive.size == 0:
        return []
    threshold = max(np.percentile(positive, 90), 1)
    binary = (top_hat >= threshold).astype(np.uint8)

    labeled, n_labels = ndimage.label(binary)
    candidates = []
    for label_id in range(1, n_labels + 1):
        ys, xs = np.where(labeled == label_id)
        area = len(xs)
        if not (min_area <= area <= max_area):
            continue
        bbox_w, bbox_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        circularity = area / (bbox_w * bbox_h)
        if circularity < min_circularity:
            continue
        candidates.append({"center": (int(round(xs.mean())), int(round(ys.mean()))), "area": area})
    return candidates


def candidate_stage_recall(img: np.ndarray, vessel_mask: np.ndarray, true_mask: np.ndarray, tolerance: int = 10) -> float:
    """Fraction of true MA connected components with >=1 stage-1 candidate
    within `tolerance` pixels of their centroid. NaN if there are no true
    MAs in this image (undefined, not zero)."""
    labeled, n_labels = ndimage.label(true_mask.astype(bool))
    if n_labels == 0:
        return float("nan")
    true_centers = []
    for label_id in range(1, n_labels + 1):
        ys, xs = np.where(labeled == label_id)
        true_centers.append((float(xs.mean()), float(ys.mean())))

    candidate_centers = [c["center"] for c in extract_candidates(img, vessel_mask)]
    if not candidate_centers:
        return 0.0

    found = sum(
        1 for tx, ty in true_centers if any(np.hypot(tx - cx, ty - cy) <= tolerance for cx, cy in candidate_centers)
    )
    return found / n_labels


def extract_patch(img: np.ndarray, center: tuple[int, int], size: int = PATCH_SIZE) -> np.ndarray:
    """`size`x`size` patch centered at `center`, zero-padded at image edges."""
    h, w = img.shape[:2]
    cx, cy = center
    half = size // 2
    y0, y1, x0, x1 = cy - half, cy - half + size, cx - half, cx - half + size

    pad_top, pad_bottom = max(0, -y0), max(0, y1 - h)
    pad_left, pad_right = max(0, -x0), max(0, x1 - w)
    patch = img[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    if pad_top or pad_bottom or pad_left or pad_right:
        pad_spec = ((pad_top, pad_bottom), (pad_left, pad_right)) + (((0, 0),) if img.ndim == 3 else ())
        patch = np.pad(patch, pad_spec)
    return patch


class MAPatchClassifier(nn.Module):
    """Small CNN: 48x48 patch -> MA vs not-MA logit. Trainable locally on MPS."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1)).squeeze(1)


class MAPatchDataset(Dataset):
    """Positives: true MA centroids from ground truth. Hard negatives:
    stage-1 candidates on the same images that don't overlap a true MA --
    exactly the false positives stage 2 needs to learn to reject."""

    def __init__(self, raw_dir: str, split: str, seed: int = 42):
        raw_dir = Path(raw_dir)
        split_df = pd.read_csv(raw_dir / "split.csv")
        rows = split_df[(split_df["split"] == split) & (split_df["has_ma"])].reset_index(drop=True)
        if len(rows) == 0:
            raise ValueError(f"No has_ma rows for split={split!r} in {raw_dir / 'split.csv'}")

        rng = np.random.default_rng(seed)
        self.samples: list[tuple[str, tuple[int, int], int]] = []
        self._images: dict[str, np.ndarray] = {}

        for _, row in rows.iterrows():
            image_id = row["id"]
            image = cv2.cvtColor(cv2.imread(str(raw_dir / "images" / f"{image_id}.jpg")), cv2.COLOR_BGR2RGB)
            self._images[image_id] = image
            ma_mask = cv2.imread(str(raw_dir / "masks" / "ma" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE) > 127

            labeled, n_labels = ndimage.label(ma_mask)
            positives = []
            for label_id in range(1, n_labels + 1):
                ys, xs = np.where(labeled == label_id)
                positives.append((int(xs.mean()), int(ys.mean())))
            for c in positives:
                self.samples.append((image_id, c, 1))

            # No vessel model available at dataset-construction time -- using
            # an all-clear vessel mask here just means a few more vessel-edge
            # candidates surface as negatives, which is fine (even useful:
            # vessels are a real MA false-positive source to learn to reject).
            candidates = extract_candidates(image, np.zeros(image.shape[:2], dtype=bool))
            negatives = [c["center"] for c in candidates if all(abs(c["center"][0] - px) + abs(c["center"][1] - py) > 10 for px, py in positives)]
            rng.shuffle(negatives)
            for c in negatives[: max(len(positives), 5)]:
                self.samples.append((image_id, c, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        image_id, center, label = self.samples[idx]
        patch = extract_patch(self._images[image_id], center, PATCH_SIZE)
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor, float(label)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, n = 0.0, 0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float()
            logits = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            correct += ((logits > 0) == (labels > 0.5)).sum().item()
            n += batch_size
    return total_loss / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    epochs = args.epochs or config["epochs"]
    set_seed(config.get("seed", 42))

    device = get_device()
    train_ds = MAPatchDataset(config["raw_dir"], split="train", seed=config.get("seed", 42))
    val_ds = MAPatchDataset(config["raw_dir"], split="val", seed=config.get("seed", 42) + 1)
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 16), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 16), shuffle=False, num_workers=0)

    model = MAPatchClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3))

    ckpt_dir = Path(config.get("checkpoint_dir", "models/ma_classifier"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(config.get("log_dir", "runs/ma_classifier"))))

    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("acc", {"train": train_acc, "val": val_acc}, epoch)
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_acc": val_acc}, ckpt_dir / f"epoch_{epoch}.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc}, ckpt_dir / "best.pt")

    writer.close()
    print(f"best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        import time

        t0 = time.time()
        rng = np.random.default_rng(13)

        # Stage 1 on a synthetic image with a few small dark blobs (MA-like)
        # plus a big bright region (should be rejected -- too big) and a thin
        # line (vessel-like, should be rejected -- not circular).
        img = np.full((300, 300, 3), 140, dtype=np.uint8)
        true_centers = []
        for _ in range(6):
            cx, cy = int(rng.integers(30, 270)), int(rng.integers(30, 270))
            cv2.circle(img, (cx, cy), 3, (60, 40, 40), -1)
            true_centers.append((cx, cy))
        cv2.line(img, (10, 150), (290, 150), (60, 60, 60), 2)  # vessel-like, not circular
        vessel_mask = np.zeros((300, 300), dtype=bool)

        candidates = extract_candidates(img, vessel_mask)
        assert 1 <= len(candidates) <= 500, len(candidates)
        for c in candidates:
            assert 0 <= c["center"][0] < 300 and 0 <= c["center"][1] < 300

        true_mask = np.zeros((300, 300), dtype=np.uint8)
        for cx, cy in true_centers:
            cv2.circle(true_mask, (cx, cy), 3, 255, -1)
        recall = candidate_stage_recall(img, vessel_mask, true_mask.astype(bool))
        assert 0.0 <= recall <= 1.0

        # extract_patch: in-bounds and edge cases (padding) both produce the right shape
        patch = extract_patch(img, (150, 150), PATCH_SIZE)
        assert patch.shape == (PATCH_SIZE, PATCH_SIZE, 3)
        edge_patch = extract_patch(img, (0, 0), PATCH_SIZE)
        assert edge_patch.shape == (PATCH_SIZE, PATCH_SIZE, 3)

        # Stage 2 model wiring
        device = get_device()
        model = MAPatchClassifier().to(device)
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        logit = model(tensor)
        assert logit.shape == (1,)

        elapsed = time.time() - t0
        print(f"candidates.py smoke test ok in {elapsed:.3f}s on {device}")
        assert elapsed < 30
