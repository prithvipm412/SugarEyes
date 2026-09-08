"""Train the DR grading model (CORN ordinal head) with class-weighted
sampling, albumentations + synthetic degradation augmentation, cosine LR,
and early stopping on validation QWK. Checkpoints every epoch and resumes
from the latest one automatically -- Kaggle sessions die at 12h without
warning (see AGENTS.md).

Real full-scale training runs on Kaggle; this also runs locally on the
small APTOS dev sample for wiring smoke tests, but a local run's QWK is not
the real Phase 2 result (see AGENTS.md: heavy grading training is Kaggle-only).
"""
import argparse
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from ..device import get_device
from ..eval.metrics import quadratic_weighted_kappa
from .dataset import GradingDataset, class_weights
from .model import GradingModel
from .ordinal import corn_loss, decode_corn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return checkpoints[-1] if checkpoints else None


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with amp_context(device):
                logits = model(images)
                loss = corn_loss(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            all_true.append(labels.cpu().numpy())
            all_pred.append(decode_corn(logits).cpu().numpy())
    qwk = quadratic_weighted_kappa(np.concatenate(all_true), np.concatenate(all_pred))
    return total_loss / n, qwk


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
    image_size = config.get("image_size", 224)
    backbone = config.get("backbone", "resnet34")
    train_ds = GradingDataset(config["manifest_csv"], split="train", image_size=image_size, train=True, seed=config.get("seed", 42))
    val_ds = GradingDataset(config["manifest_csv"], split="val", image_size=image_size, train=False)

    sampler = WeightedRandomSampler(class_weights(train_ds.rows), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 16), sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 16), shuffle=False, num_workers=0)

    model = GradingModel(backbone_name=backbone).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4))

    ckpt_dir = Path(config.get("checkpoint_dir", "models/grading"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(config.get("log_dir", "runs/grading"))))

    start_epoch, best_qwk, epochs_without_improvement = 1, -1.0, 0
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is not None:
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"resumed from {latest_ckpt} at epoch {start_epoch}")
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            best_qwk = torch.load(best_path, map_location=device)["val_qwk"]

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, last_epoch=start_epoch - 2)
    patience = config.get("patience", 5)

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_qwk = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_qwk = run_epoch(model, val_loader, device)
        scheduler.step()
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("qwk", {"train": train_qwk, "val": val_qwk}, epoch)
        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_qwk={train_qwk:.4f}  "
            f"val_loss={val_loss:.4f} val_qwk={val_qwk:.4f}"
        )

        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_qwk": val_qwk, "backbone": backbone},
            ckpt_dir / f"epoch_{epoch}.pt",
        )

        if val_qwk > best_qwk:
            best_qwk, epochs_without_improvement = val_qwk, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_qwk": val_qwk, "backbone": backbone}, ckpt_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping at epoch {epoch} (no val_qwk improvement for {patience} epochs)")
                break

    writer.close()
    print(f"best val_qwk={best_qwk:.4f}")


if __name__ == "__main__":
    main()
