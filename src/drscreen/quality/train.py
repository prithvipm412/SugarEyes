"""Train the IQA model (GRADEABLE / ENHANCEABLE / REJECT) on synthetic
severity-derived labels. See dataset.py for the labeling scheme and
AGENTS.md for why real EyeQ labels are Kaggle-only.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..device import get_device
from .dataset import SEVERITY_BANDS, IQADataset
from .model import CLASS_NAMES, IQAModel, compute_feature_stats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, device, criterion, optimizer=None) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, features, labels in loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            logits = model(images, features)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def save_gradeable_feature_stats(train_ds: IQADataset, out_path: Path) -> None:
    """Feature stats over GRADEABLE (label 0) training samples, used at
    inference time by quality.model.reject_reason to explain a REJECT verdict."""
    gradeable_features = []
    for row_idx in range(len(train_ds.rows)):
        idx = row_idx * len(SEVERITY_BANDS)  # label 0 (GRADEABLE) is the first item of each block
        _tensor, features, label = train_ds[idx]
        assert label == 0
        gradeable_features.append(features.numpy())
    stats = compute_feature_stats(np.stack(gradeable_features))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)


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
    train_ds = IQADataset(config["manifest_csv"], split="train", image_size=image_size)
    val_ds = IQADataset(config["manifest_csv"], split="val", image_size=image_size)
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 16), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 16), shuffle=False, num_workers=0)

    model = IQAModel(backbone_name=config.get("backbone", "resnet18")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4))
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(config.get("checkpoint_dir", "models/iqa"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(config.get("log_dir", "runs/iqa"))))

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("acc", {"train": train_acc, "val": val_acc}, epoch)
        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Checkpoint every epoch -- Kaggle-style sessions can die without warning,
        # and this also protects the local MPS run against interruption.
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_acc": val_acc},
            ckpt_dir / f"epoch_{epoch}.pt",
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc}, ckpt_dir / "best.pt")

    save_gradeable_feature_stats(train_ds, ckpt_dir / "feature_stats.json")
    writer.close()
    print(f"best val_acc={best_val_acc:.4f} classes={CLASS_NAMES}")


if __name__ == "__main__":
    main()
