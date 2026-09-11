"""Vessel segmentation: a lightweight U-Net trained on 48x48 patches from
DRIVE + CHASE_DB1 + STARE, plus vessel_metrics() -- tortuosity, branching
density, calibre -- computed from any binary vessel mask (real or predicted).

Patch training turns ~88 images into ~200k patches (see AGENTS.md) -- this is
what makes training work at all on a dataset this small. Train/val is split
at the *dataset* level (e.g. train on DRIVE+CHASE_DB1, validate on STARE),
not by randomly splitting patches, so no patch from a validation image can
leak into training.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy import ndimage
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from ..device import get_device

PATCH_SIZE = 48


class VesselModel(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VesselPatchDataset(Dataset):
    def __init__(self, raw_dir: str, dataset_names: list[str], patches_per_image: int = 400, fg_fraction: float = 0.5, seed: int = 42):
        self.pairs = []
        for name in dataset_names:
            images_dir = Path(raw_dir) / name / "images"
            masks_dir = Path(raw_dir) / name / "masks"
            for img_path in sorted(images_dir.glob("*.png")):
                mask_path = masks_dir / img_path.name
                if mask_path.exists():
                    self.pairs.append((img_path, mask_path))
        if not self.pairs:
            raise ValueError(f"No image/mask pairs found under {raw_dir} for datasets {dataset_names}")

        self.patches_per_image = patches_per_image
        self.fg_fraction = fg_fraction
        self.rng = np.random.default_rng(seed)
        # ~88 full-resolution retina images is a few hundred MB at most --
        # well within the 8GB local budget (see AGENTS.md) -- so caching in
        # memory avoids re-decoding the same handful of images 200k times.
        self._cache = [
            (cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB), cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127)
            for ip, mp in self.pairs
        ]

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._cache[idx % len(self.pairs)]
        h, w = mask.shape
        want_fg = self.rng.random() < self.fg_fraction
        fg_ys, fg_xs = np.where(mask)

        y0, x0 = 0, 0
        for _ in range(20):
            if want_fg and len(fg_ys) > 0:
                pick = self.rng.integers(0, len(fg_ys))
                cy, cx = int(fg_ys[pick]), int(fg_xs[pick])
            else:
                cy, cx = int(self.rng.integers(0, h)), int(self.rng.integers(0, w))
            y0, x0 = cy - PATCH_SIZE // 2, cx - PATCH_SIZE // 2
            if 0 <= y0 and y0 + PATCH_SIZE <= h and 0 <= x0 and x0 + PATCH_SIZE <= w:
                break
        y0 = int(np.clip(y0, 0, h - PATCH_SIZE))
        x0 = int(np.clip(x0, 0, w - PATCH_SIZE))

        img_patch = image[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
        mask_patch = mask[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
        tensor = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        target = torch.from_numpy(mask_patch.astype(np.float32)).unsqueeze(0)
        return tensor, target


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item()
    return (2 * intersection + eps) / (union + eps)


def vessel_metrics(mask: np.ndarray) -> dict:
    """Tortuosity, branching density, and mean calibre from a binary vessel
    mask, via skeletonization. These are approximate, cheap proxies (not a
    research-grade vessel tracer) meant to feed the neovascularisation
    heuristic and the fusion model's structured features."""
    binary = mask.astype(bool)
    if binary.sum() == 0:
        return {"tortuosity": 0.0, "branching_density": 0.0, "mean_calibre": 0.0}

    skeleton = skeletonize(binary)
    distance = cv2.distanceTransform((binary * 255).astype(np.uint8), cv2.DIST_L2, 5)
    calibre_samples = distance[skeleton] * 2  # local vessel diameter at skeleton points
    mean_calibre = float(calibre_samples.mean()) if calibre_samples.size > 0 else 0.0

    # 8-neighbour count via explicit shifting -- cv2.filter2D with a uint8
    # "ring" kernel (ones except a zero center) silently returns wrong
    # values on some OpenCV builds (verified: saturates to 0/1 instead of
    # summing), so this is done with plain numpy instead.
    padded = np.pad(skeleton.astype(np.uint8), 1, mode="constant")
    neighbour_count = sum(
        padded[dy : dy + skeleton.shape[0], dx : dx + skeleton.shape[1]]
        for dy in range(3)
        for dx in range(3)
        if not (dy == 1 and dx == 1)
    )
    branch_points = int(((neighbour_count >= 3) & skeleton).sum())
    skeleton_length = int(skeleton.sum())
    branching_density = branch_points / skeleton_length if skeleton_length > 0 else 0.0

    # Arc-chord tortuosity per connected skeleton component: path length
    # (pixel count) over a chord-length proxy (2x max distance from the
    # component's centroid). An approximation of the textbook per-segment
    # measure, adequate for a scalar feature rather than a diagnostic trace.
    labeled, n_labels = ndimage.label(skeleton)
    tortuosities = []
    for label_id in range(1, n_labels + 1):
        ys, xs = np.where(labeled == label_id)
        if len(ys) < 5:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        centroid = pts.mean(axis=0)
        chord_length = 2 * np.linalg.norm(pts - centroid, axis=1).max()
        if chord_length > 0:
            tortuosities.append(len(ys) / chord_length)
    tortuosity = float(np.mean(tortuosities)) if tortuosities else 0.0

    return {"tortuosity": tortuosity, "branching_density": float(branching_density), "mean_calibre": mean_calibre}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_dice, n = 0.0, 0.0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = combined_loss(logits, targets)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_dice += dice_score(logits, targets) * batch_size
            n += batch_size
    return total_loss / n, total_dice / n


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
    train_ds = VesselPatchDataset(
        "data/raw/vessels", config.get("train_datasets", ["drive", "chasedb1"]),
        patches_per_image=config.get("patches_per_image", 400), seed=config.get("seed", 42),
    )
    val_ds = VesselPatchDataset(
        "data/raw/vessels", config.get("val_datasets", ["stare"]),
        patches_per_image=config.get("val_patches_per_image", 100), seed=config.get("seed", 42) + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 64), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 64), shuffle=False, num_workers=0)

    model = VesselModel(encoder_name=config.get("encoder", "resnet18")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3))

    ckpt_dir = Path(config.get("checkpoint_dir", "models/vessels"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(config.get("log_dir", "runs/vessels"))))

    best_dice = -1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, device)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("dice", {"train": train_dice, "val": val_dice}, epoch)
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_dice={train_dice:.4f}  val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

        encoder = config.get("encoder", "resnet18")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_dice": val_dice, "encoder": encoder}, ckpt_dir / f"epoch_{epoch}.pt")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_dice": val_dice, "encoder": encoder}, ckpt_dir / "best.pt")

    writer.close()
    print(f"best val_dice={best_dice:.4f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        import tempfile
        import time

        t0 = time.time()
        rng = np.random.default_rng(9)

        # vessel_metrics on a hand-constructed pattern: a single straight
        # line (tortuosity ~= 1, no branch points) vs. a Y-branch (has one).
        straight = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(straight, (10, 50), (90, 50), 1, 1)
        m = vessel_metrics(straight.astype(bool))
        assert 0.9 <= m["tortuosity"] <= 1.3, m
        assert m["branching_density"] == 0.0, m

        branch = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(branch, (50, 50), (10, 10), 1, 1)
        cv2.line(branch, (50, 50), (10, 90), 1, 1)
        cv2.line(branch, (50, 50), (90, 50), 1, 1)
        m = vessel_metrics(branch.astype(bool))
        assert m["branching_density"] > 0, m

        empty = np.zeros((100, 100), dtype=bool)
        m = vessel_metrics(empty)
        assert m == {"tortuosity": 0.0, "branching_density": 0.0, "mean_calibre": 0.0}

        # model + patch dataset wiring, on 5 synthetic image/mask pairs
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in ["synth_a", "synth_b"]:
                (tmp_path / name / "images").mkdir(parents=True)
                (tmp_path / name / "masks").mkdir(parents=True)
                for i in range(3):
                    img = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
                    mask = np.zeros((120, 120), dtype=np.uint8)
                    cv2.line(mask, (rng.integers(0, 120), rng.integers(0, 120)), (rng.integers(0, 120), rng.integers(0, 120)), 255, 2)
                    cv2.imwrite(str(tmp_path / name / "images" / f"img_{i}.png"), img)
                    cv2.imwrite(str(tmp_path / name / "masks" / f"img_{i}.png"), mask)

            ds = VesselPatchDataset(str(tmp_path), ["synth_a"], patches_per_image=5)
            assert len(ds) == 15
            device = get_device()
            model = VesselModel(pretrained=False).to(device)
            images = torch.stack([ds[i][0] for i in range(5)]).to(device)
            targets = torch.stack([ds[i][1] for i in range(5)]).to(device)
            logits = model(images)
            assert logits.shape == targets.shape == (5, 1, PATCH_SIZE, PATCH_SIZE)
            loss = combined_loss(logits, targets)
            assert torch.isfinite(loss)
            score = dice_score(logits, targets)
            assert 0.0 <= score <= 1.0

        elapsed = time.time() - t0
        print(f"vessels.py smoke test ok in {elapsed:.3f}s on {device}")
        assert elapsed < 30
