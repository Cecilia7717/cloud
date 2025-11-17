# train_pure_oom_safe.py
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, default_collate

# ----------------------------------------------------------------------
# Your project utilities (keep them exactly as before)
# ----------------------------------------------------------------------
from utils.utils import get_transform, get_metrics, get_train_test_datasets, get_model
from utils.loss import TrainingLoss
from utils.quant_utils.export.export_qonnx import export_model_qonnx

# ----------------------------------------------------------------------
# 1. Average meter
# ----------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += float(val) * int(n)
        self.cnt += int(n)

    @property
    def avg(self):
        return self.sum / max(1, self.cnt)


# ----------------------------------------------------------------------
# 2. CPU-only collate + transform
# ----------------------------------------------------------------------
def make_cpu_collate(transform):
    """Apply per-sample transform on CPU, then stack once."""
    def collate(batch):
        imgs, masks = [], []
        for img, mask in batch:                     # raw samples (numpy / PIL)
            i, m = transform((img, mask))           # <-- your Albumentations / torchvision
            imgs.append(i)
            masks.append(m)
        return torch.stack(imgs), torch.stack(masks)
    return collate


# ----------------------------------------------------------------------
# 3. Train / validation steps (no manual stacking)
# ----------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    with_amp: bool,
) -> float:
    model.train()
    loss_meter = AverageMeter()

    for x, y in loader:                                 # already stacked on CPU
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=with_amp):
            pred = model(x)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), x.size(0))

        # free GPU memory ASAP
        del x, y, pred, loss

    return loss_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: dict,
    with_amp: bool,
) -> Tuple[float, dict]:
    model.eval()
    for m in metrics.values():
        m.reset()

    loss_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # loss uses one-hot (B,2,H,W)
        y_loss = y
        # metrics expect class-index (B,H,W)
        y_metric = torch.argmax(y, dim=1) if y.dim() == 4 and y.size(1) == 2 else y

        with autocast(device_type="cuda", enabled=with_amp):
            pred = model(x)
            loss = criterion(pred, y_loss)

        loss_meter.update(loss.item(), x.size(0))

        for m in metrics.values():
            m.update((pred, y_metric))

        del x, y, pred, loss, y_loss, y_metric

    results = {name: metric.compute() for name, metric in metrics.items()}
    return loss_meter.avg, results


# ----------------------------------------------------------------------
# 4. Checkpoint handling
# ----------------------------------------------------------------------
def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, best_f1):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
        },
        path,
    )


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main() -> None:
    # -------------------------- arguments --------------------------
    parser = argparse.ArgumentParser(description="Pure-PyTorch training (OOM-safe)")
    parser.add_argument("--data_path", type=str, default="/data")
    parser.add_argument("--csv_train", type=str, default="/pvc/train.csv")
    parser.add_argument("--csv_test", type=str, default="/pvc/valid.csv")
    parser.add_argument("--output_path", type=str, default="./output_pure/")
    parser.add_argument("--model", type=str, default="ags_tiny_unet_100k")
    parser.add_argument("--quant_config", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=4)      # start small
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=2)    # 2-4 is safe
    parser.add_argument("--with_amp", action="store_true")
    parser.add_argument("--input_size", nargs=2, type=int, default=[512, 512])

    parser.add_argument("--resume", type=str, default=None)

    # Ignite-like extras
    parser.add_argument("--validate_every", type=int, default=3)
    parser.add_argument("--num_warmup_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()

    # -------------------------- seed --------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------- device & output dir --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_root = Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / f"{args.model}_pure_{now}"
    run_dir.mkdir(exist_ok=True)
    print(f"Run directory: {run_dir}")

    # -------------------------- datasets & transforms --------------------------
    train_dataset, test_dataset = get_train_test_datasets(
        args.data_path,
        csv_paths={"train": args.csv_train, "test": args.csv_test},
    )

    train_transform, test_transform = get_transform(
        mean=[0.0, 0.0, 0.0], std=[8657.0, 8657.0, 8657.0]
    )

    # -------------------------- data loaders (CPU collate) --------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,                 # OOM killer
        prefetch_factor=2,                # tiny prefetch
        collate_fn=make_cpu_collate(train_transform),
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2,
        collate_fn=make_cpu_collate(test_transform),
        persistent_workers=False,
    )

    # -------------------------- model / loss / optim / scheduler --------------------------
    model = get_model(args.model, args.quant_config).to(device)
    criterion = TrainingLoss([0.1, 0.9]).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Piecewise-linear LR with warm-up (exactly like Ignite)
    total_iters = len(train_loader) * args.epochs
    warmup_iters = len(train_loader) * args.num_warmup_epochs

    def lr_lambda(step):
        if step < warmup_iters:
            return step / max(1, warmup_iters)
        progress = (step - warmup_iters) / (total_iters - warmup_iters)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler(enabled=args.with_amp)

    # Metrics (drop the loss metric – we track it ourselves)
    metrics = get_metrics(class_count=2, criterion=criterion)
    metrics.pop("Loss", None)

    # -------------------------- resume --------------------------
    start_epoch = 1
    best_f1 = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", -1.0)
        print(f"Resumed from epoch {start_epoch-1}")

    # -------------------------- TensorBoard (optional) --------------------------
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=run_dir / "tb")
    except Exception:
        tb_writer = None

    # -------------------------- training loop --------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, args.with_amp
        )
        print(f"Train loss: {train_loss:.5f}")

        # ---------- validation (only when requested) ----------
        val_loss, val_metrics = None, None
        do_val = (epoch % args.validate_every == 0) or (epoch == args.epochs)
        if do_val:
            val_loss, val_metrics = validate(
                model, test_loader, criterion, device,
                metrics, args.with_amp
            )
            f1 = float(val_metrics.get("F1", 0.0))
            print(f"Val loss: {val_loss:.5f} | F1: {f1:.5f}")

        # ---------- LR step (per-batch via LambdaLR) ----------
        scheduler.step()

        # ---------- TensorBoard ----------
        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            if val_loss is not None:
                tb_writer.add_scalar("Loss/val", val_loss, epoch)
                tb_writer.add_scalar("F1/val", val_metrics.get("F1", 0.0), epoch)

        # ---------- checkpoint (latest) ----------
        save_checkpoint(run_dir / "latest.pt", model, optimizer,
                        scheduler, epoch, best_f1)

        # ---------- best model + QONNX ----------
        if do_val and f1 > best_f1:
            best_f1 = f1
            save_checkpoint(run_dir / "best.pt", model, optimizer,
                            scheduler, epoch, best_f1)
            print(f"New best F1 {best_f1:.5f} @ epoch {epoch}")

            input_shape = (1, 3, args.input_size[0], args.input_size[1])
            qonnx_path = run_dir / "best_qonnx.onnx"
            print(f"Exporting QONNX to {qonnx_path}")
            export_model_qonnx(model, qonnx_path, input_shape=input_shape)

        # clean GPU memory between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------- finish --------------------------
    if tb_writer:
        tb_writer.close()

    print("\nTraining finished!")
    print(f"Best model : {run_dir / 'best.pt'}")
    print(f"QONNX model: {run_dir / 'best_qonnx.onnx'}")


if __name__ == "__main__":
    main()