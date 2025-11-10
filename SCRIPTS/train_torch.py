import os
import time
from pathlib import Path
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import numpy as np

# === your existing utilities ===
from utils.utils import get_transform, get_metrics, get_train_test_datasets, get_model
from utils.loss import TrainingLoss
from utils.quant_utils.export.export_qonnx import export_model_qonnx


# ------------------------------------------------------------
# Utility: Metric accumulation
# ------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


# ------------------------------------------------------------
# Training Step
# ------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, transform, scaler, with_amp):
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        # Apply transforms like your Ignite pipeline
        new_x, new_y = [], []
        for i in range(len(x)):
            sample_img, sample_mask = transform((x[i], y[i]))
            new_x.append(sample_img)
            new_y.append(sample_mask)
        x = torch.stack(new_x).to(device)
        y = torch.stack(new_y).to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=with_amp):
            pred = model(x)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), x.size(0))

    return loss_meter.avg


# ------------------------------------------------------------
# Validation Step
# ------------------------------------------------------------
def validate(model, loader, criterion, device, transform, metrics, with_amp):
    model.eval()

    for metric in metrics.values():
        metric.reset()

    loss_meter = AverageMeter()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Apply transform
            new_x, new_y = [], []
            for i in range(len(x)):
                sample_img, sample_mask = transform((x[i], y[i]))
                new_x.append(sample_img)
                new_y.append(sample_mask)
            x = torch.stack(new_x).to(device)
            y = torch.stack(new_y).to(device)
            # Convert mask to single-channel class indices if needed
            # --- keep original y for loss ---
            y_loss = y    # shape (B,2,H,W)

            # --- create metric-friendly index mask ---
            if y.dim() == 4 and y.size(1) == 2:
                y_metric = torch.argmax(y, dim=1)   # (B,H,W)
            else:
                y_metric = y

            with autocast(device_type="cuda", enabled=with_amp):
                pred = model(x)
                loss = criterion(pred, y_loss)

            loss_meter.update(loss.item(), x.size(0))

            # --- update metrics with index mask ---
            for name, metric in metrics.items():
                metric.update((pred, y_metric))


    # Compute metric values
    metric_results = {name: metric.compute() for name, metric in metrics.items()}
    return loss_meter.avg, metric_results


# ------------------------------------------------------------
# Save checkpoint
# ------------------------------------------------------------
def save_checkpoint(path, model, optimizer, epoch, best_score):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    torch.save(checkpoint, path)


# ------------------------------------------------------------
# Main Training Function
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data")
    parser.add_argument("--csv_train", type=str, default="/pvc/train.csv")
    parser.add_argument("--csv_test", type=str, default="/pvc/valid.csv")
    parser.add_argument("--output_path", type=str, default="./output_pure/")
    parser.add_argument("--model", type=str, default="ags_tiny_unet_100k")
    parser.add_argument("--quant_config", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--with_amp", action="store_true")
    parser.add_argument("--input_size", nargs=2, type=int, default=[512, 512])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # -------------------------
    # Setup
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # timestamped output
    out_dir = Path(args.output_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    # -------------------------
    # Dataset
    # -------------------------
    train_dataset, test_dataset = get_train_test_datasets(
        args.data_path,
        csv_paths={"train": args.csv_train, "test": args.csv_test},
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=2 * args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # transforms
    train_transform, test_transform = get_transform(mean=[0,0,0], std=[8657,8657,8657])

    # -------------------------
    # Model / Loss / Optimizer
    # -------------------------
    model = get_model(args.model, args.quant_config)
    model = model.to(device)

    criterion = TrainingLoss([0.1, 0.9]).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    scaler = GradScaler(enabled=args.with_amp)

    # -------------------------
    # Metrics
    # -------------------------
    metrics = get_metrics(class_count=2, criterion=criterion)

    if "Loss" in metrics:
        del metrics["Loss"]

    # -------------------------
    # Resume
    # -------------------------
    start_epoch = 1
    best_f1 = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_score", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, criterion,
            optimizer, device, train_transform,
            scaler, args.with_amp
        )
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_metrics = validate(
            model, test_loader, criterion,
            device, test_transform, metrics,
            args.with_amp
        )

        f1 = float(val_metrics.get("F1", 0))
        print(f"Val Loss: {val_loss:.4f}, F1: {f1:.4f}")
        print("Metrics:", val_metrics)

        scheduler.step()

        # Save latest
        save_checkpoint(out_dir / "latest.pt", model, optimizer, epoch, best_f1)

        # Save best
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_f1)

            # also export QONNX (equivalent to Ignite handler)
            print("Exporting QONNX...")
            input_shape = (1, 3, args.input_size[0], args.input_size[1])
            export_model_qonnx(model, out_dir / "best_qonnx.onnx", input_shape=input_shape)

    print("Training completed.")


if __name__ == "__main__":
    main()
