"""
Training loop for the CNN + Bi-GRU + Attention ECG classifier.

Training protocol (as described in the report):
    optimizer  : Adam, initial LR 1e-4
    batch size : 32
    epochs     : up to 100 with early stopping (patience 10 on val loss)
    regularize : dropout 0.3 (in the model) + LR halved every 10 epochs
    imbalance  : optional cost-sensitive (class-weighted) cross-entropy

Run:
    python src/train.py --data_dir data/processed --epochs 100 --batch_size 32 --lr 1e-4
"""
from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ECGWindowDataset, class_weights
from model import ECGArrhythmiaNet


def load_split(data_dir: str) -> tuple:
    """Load preprocessed windows and return (X_train, y_train, X_val, y_val).

    Implement this for your saved layout, e.g. np.load(...) of arrays written by
    preprocessing.py. X_* are float32 (N, 12, 5000); y_* are int labels in [0, 8].
    """
    raise NotImplementedError(
        f"Implement load_split() for the preprocessed data in {data_dir!r}."
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--weighted_loss", action="store_true", help="cost-sensitive CE for imbalance")
    ap.add_argument("--out", default="checkpoints/best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_val, y_val = load_split(args.data_dir)
    train_ds = ECGWindowDataset(X_train, y_train)
    val_ds = ECGWindowDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = ECGArrhythmiaNet().to(device)
    weight = class_weights(y_train).to(device) if args.weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # LR halved every 10 epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"[{epoch:3d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_loss < best_val:                     # early stopping on val loss
            best_val, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f}).")
                break

    if best_state is not None:
        torch.save(best_state, args.out)
        print(f"Saved best checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
