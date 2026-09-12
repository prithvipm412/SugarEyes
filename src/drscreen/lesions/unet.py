"""4-class lesion segmentation: microaneurysms, haemorrhages, hard exudates,
soft exudates. One U-Net, 384x384 patches randomly cropped from IDRiD's
full-resolution images (patch-based sampling turns 81 images into enough
training signal, and preserves pixel-level lesion detail that resizing the
whole image down would destroy).

Soft exudates are present in only about half of IDRiD's training images
(see AGENTS.md), and microaneurysms are sub-pixel-scale even at this patch
size -- both are expected to score poorly here. That's reported per-channel,
honestly, never hidden. (candidates.py's dedicated MA cascade is what
actually handles microaneurysm detection well; this U-Net's MA channel is a
coarser signal alongside it, not a replacement.)
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from ..device import get_device

PATCH_SIZE = 384
LESION_CLASSES = ["ma", "he", "ex", "se"]
# Per-channel loss weighting: soft exudates are rare, so upweight them or
# the network just learns to always predict "absent" for that channel.
CLASS_WEIGHTS = {"ma": 1.0, "he": 1.0, "ex": 1.0, "se": 3.0}


class LesionUNet(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=len(LESION_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LesionPatchDataset(Dataset):
    def __init__(self, raw_dir: str, split: str, patches_per_image: int = 50, fg_fraction: float = 0.7, seed: int = 42):
        raw_dir = Path(raw_dir)
        split_df = pd.read_csv(raw_dir / "split.csv")
        self.rows = split_df[split_df["split"] == split].reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError(f"No rows for split={split!r} in {raw_dir / 'split.csv'}")
        self.raw_dir = raw_dir
        self.patches_per_image = patches_per_image
        self.fg_fraction = fg_fraction
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows) * self.patches_per_image

    def _load(self, image_id: str) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.cvtColor(cv2.imread(str(self.raw_dir / "images" / f"{image_id}.jpg")), cv2.COLOR_BGR2RGB)
        masks = []
        for cls in LESION_CLASSES:
            mask_path = self.raw_dir / "masks" / cls / f"{image_id}.png"
            if mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
            else:
                m = np.zeros(image.shape[:2], dtype=bool)
            masks.append(m)
        return image, np.stack(masks, axis=0)  # (4, H, W)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.rows.loc[idx % len(self.rows), "id"]
        image, masks = self._load(image_id)
        h, w = image.shape[:2]

        any_lesion = masks.any(axis=0)
        fg_ys, fg_xs = np.where(any_lesion)
        want_fg = self.rng.random() < self.fg_fraction

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
        y0 = int(np.clip(y0, 0, max(h - PATCH_SIZE, 0)))
        x0 = int(np.clip(x0, 0, max(w - PATCH_SIZE, 0)))

        img_patch = image[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
        mask_patch = masks[:, y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]

        pad_h = PATCH_SIZE - img_patch.shape[0]
        pad_w = PATCH_SIZE - img_patch.shape[1]
        if pad_h > 0 or pad_w > 0:
            img_patch = np.pad(img_patch, ((0, pad_h), (0, pad_w), (0, 0)))
            mask_patch = np.pad(mask_patch, ((0, 0), (0, pad_h), (0, pad_w)))

        tensor = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        target = torch.from_numpy(mask_patch.astype(np.float32))
        return tensor, target


def per_channel_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    return (2 * intersection + eps) / (union + eps)


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([CLASS_WEIGHTS[c] for c in LESION_CLASSES], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=(0, 2, 3))
    dice = 1 - per_channel_dice(logits, targets)
    return ((bce + dice) * weights).sum() / weights.sum()


def per_channel_dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> np.ndarray:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(0, 2, 3))
    union = preds.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).detach().cpu().numpy()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n = 0.0, 0
    dice_sums = np.zeros(len(LESION_CLASSES))
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
            dice_sums += per_channel_dice_score(logits, targets) * batch_size
            n += batch_size
    return total_loss / n, dice_sums / n


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
    train_ds = LesionPatchDataset(config["raw_dir"], split="train", patches_per_image=config.get("patches_per_image", 50), seed=config.get("seed", 42))
    val_ds = LesionPatchDataset(config["raw_dir"], split="val", patches_per_image=config.get("val_patches_per_image", 20), seed=config.get("seed", 42) + 1)
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 4), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 4), shuffle=False, num_workers=0)

    encoder = config.get("encoder", "resnet34")
    model = LesionUNet(encoder_name=encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4))

    ckpt_dir = Path(config.get("checkpoint_dir", "models/lesion_unet"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(config.get("log_dir", "runs/lesion_unet"))))

    start_epoch, best_mean_dice, val_dice = 1, -1.0, np.zeros(len(LESION_CLASSES))
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if checkpoints:
        checkpoint = torch.load(checkpoints[-1], map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"resumed from {checkpoints[-1]} at epoch {start_epoch}", flush=True)
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            best_mean_dice = float(np.mean(torch.load(best_path, map_location=device)["val_dice"]))

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, device)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        for i, cls in enumerate(LESION_CLASSES):
            writer.add_scalars(f"dice_{cls}", {"train": train_dice[i], "val": val_dice[i]}, epoch)

        dice_str = "  ".join(f"{c}={val_dice[i]:.3f}" for i, c in enumerate(LESION_CLASSES))
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_dice[{dice_str}]", flush=True)

        mean_dice = float(val_dice.mean())
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_dice": val_dice.tolist(), "encoder": encoder}, ckpt_dir / f"epoch_{epoch}.pt")
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_dice": val_dice.tolist(), "encoder": encoder}, ckpt_dir / "best.pt")

    writer.close()
    # Re-load best.pt rather than reusing the loop's last `val_dice` --
    # that's whichever epoch the loop happened to end on, not necessarily
    # the epoch that achieved best_mean_dice, so pairing them here would
    # print a mean and a per-class breakdown from two different epochs.
    best_checkpoint = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    print(f"best mean val_dice={best_mean_dice:.4f}  per-class={dict(zip(LESION_CLASSES, best_checkpoint['val_dice']))}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        import tempfile
        import time

        t0 = time.time()
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "images").mkdir()
            rows = []
            for i in range(5):
                image_id = f"synth_{i}"
                img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
                cv2.imwrite(str(tmp_path / "images" / f"{image_id}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                has_masks = {}
                for cls in LESION_CLASSES:
                    (tmp_path / "masks" / cls).mkdir(parents=True, exist_ok=True)
                    if cls != "se" or i % 2 == 0:  # mimic SE being rare/absent for some images
                        mask = np.zeros((200, 200), dtype=np.uint8)
                        cv2.circle(mask, (int(rng.integers(20, 180)), int(rng.integers(20, 180))), 8, 255, -1)
                        cv2.imwrite(str(tmp_path / "masks" / cls / f"{image_id}.png"), mask)
                        has_masks[f"has_{cls}"] = True
                    else:
                        has_masks[f"has_{cls}"] = False
                rows.append({"id": image_id, "split": "train", **has_masks})
            pd.DataFrame(rows).to_csv(tmp_path / "split.csv", index=False)

            ds = LesionPatchDataset(str(tmp_path), split="train", patches_per_image=3)
            assert len(ds) == 15
            device = get_device()
            model = LesionUNet(pretrained=False).to(device)
            images = torch.stack([ds[i][0] for i in range(4)]).to(device)
            targets = torch.stack([ds[i][1] for i in range(4)]).to(device)
            assert images.shape == (4, 3, PATCH_SIZE, PATCH_SIZE)
            assert targets.shape == (4, len(LESION_CLASSES), PATCH_SIZE, PATCH_SIZE)

            logits = model(images)
            loss = combined_loss(logits, targets)
            assert torch.isfinite(loss)
            dice = per_channel_dice_score(logits, targets)
            assert dice.shape == (len(LESION_CLASSES),)
            assert np.isfinite(dice).all()

        elapsed = time.time() - t0
        print(f"unet.py smoke test ok in {elapsed:.3f}s on {device}")
        assert elapsed < 30
