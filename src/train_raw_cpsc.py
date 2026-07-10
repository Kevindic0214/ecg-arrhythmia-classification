"""
Train the CNN + Bi-GRU + Attention model directly on the raw CPSC-2018 release
(original `.mat` files holding an `ECG` struct + a `REFERENCE.csv` of 1-9 labels),
end to end: load -> preprocess -> train -> evaluate -> save figures/checkpoint.

This is the script used to reproduce the reported results on the real data. On an
Apple-Silicon Mac it uses the MPS (Metal) GPU automatically; otherwise CUDA or CPU.

Run:
    python src/train_raw_cpsc.py --data_dir /path/to/CPSC2018 --epochs 100
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # route MPS-unsupported ops (e.g. GRU) to CPU

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CLASS_NAMES, ECGWindowDataset, class_weights
from model import ECGArrhythmiaNet, count_parameters
from preprocessing import preprocess_recording


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_labels(data_dir: str) -> dict:
    """Union every REFERENCE.csv under data_dir -> {record: class_idx in [0,8]}."""
    ref = {}
    for path in glob.glob(os.path.join(data_dir, "**", "REFERENCE.csv"), recursive=True):
        for row in csv.reader(open(path)):
            if len(row) >= 2:
                try:
                    ref[row[0].strip()] = int(row[1]) - 1  # CPSC labels are 1-indexed
                except ValueError:
                    pass  # header row
    return ref


def load_signal(mat_path: str) -> np.ndarray:
    """Return a (12, T) array from either the original CPSC `ECG` struct or WFDB `val`."""
    m = loadmat(mat_path)
    if "ECG" in m:
        return np.asarray(m["ECG"][0][0][2], dtype=np.float64)   # struct field order: sex, age, data
    if "val" in m:
        return m["val"].astype(np.float64) / 1000.0
    raise ValueError(f"unrecognized .mat layout in {mat_path}")


def build_dataset(data_dir: str, max_records: int, seed: int):
    ref = load_labels(data_dir)
    mats = {}
    for p in glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True):
        b = os.path.basename(p)[:-4]
        if b in ref:
            mats.setdefault(b, p)
    recs = sorted(mats)
    np.random.default_rng(seed).shuffle(recs)
    recs = recs[:max_records]

    X, y = [], []
    for k, r in enumerate(recs):
        try:
            sig = load_signal(mats[r])
        except Exception:
            continue
        if sig.shape[0] != 12:
            sig = sig.T
        for w in preprocess_recording(sig):
            X.append(w.astype(np.float32)); y.append(ref[r])
        if (k + 1) % 800 == 0:
            print(f"  preprocessed {k + 1}/{len(recs)} records", flush=True)
    return np.stack(X), np.array(y, np.int64), len(recs)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot = correct = n = 0.0
    with torch.set_grad_enabled(train):
        for x, yb in loader:
            x, yb = x.to(device), yb.to(device)
            logits = model(x)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return tot / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with CPSC-2018 .mat files + REFERENCE.csv")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_records", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assets_dir", default=os.path.join(os.path.dirname(__file__), "..", "assets"))
    ap.add_argument("--out", default="checkpoints/real_model.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device()
    print("device:", device)

    X, y, n_recs = build_dataset(args.data_dir, args.max_records, args.seed)
    print(f"records {n_recs} -> windows {X.shape}")
    for c in range(9):
        print(f"  {CLASS_NAMES[c]:7s}: {(y == c).sum()}")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    tr = DataLoader(ECGWindowDataset(Xtr, ytr), batch_size=32, shuffle=True)
    va = DataLoader(ECGWindowDataset(Xva, yva), batch_size=64)

    model = ECGArrhythmiaNet().to(device)
    print("trainable parameters:", count_parameters(model))
    criterion = nn.CrossEntropyLoss(weight=class_weights(ytr).to(device))   # cost-sensitive
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    hist = {"tr_loss": [], "va_loss": [], "va_acc": []}
    best, best_state, wait = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        trl, tra = run_epoch(model, tr, criterion, device, optimizer)
        val, vac = run_epoch(model, va, criterion, device)
        scheduler.step()
        hist["tr_loss"].append(trl); hist["va_loss"].append(val); hist["va_acc"].append(vac)
        print(f"[{ep:3d}/{args.epochs}] train_loss {trl:.4f} acc {tra:.3f} | val_loss {val:.4f} acc {vac:.3f}", flush=True)
        if val < best:
            best, best_state, wait = val, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {ep} (best val_loss {best:.4f})")
                break
    model.load_state_dict(best_state)

    # evaluation
    model.eval(); probs, tgts = [], []
    with torch.no_grad():
        for xb, yb in va:
            probs.append(torch.softmax(model(xb.to(device)), 1).cpu().numpy()); tgts.append(yb.numpy())
    probs = np.concatenate(probs); tgts = np.concatenate(tgts); preds = probs.argmax(1)
    present = sorted(set(tgts.tolist()))
    acc = float(accuracy_score(tgts, preds))
    f1 = float(f1_score(tgts, preds, average="macro", labels=present))
    try:
        auc = float(roc_auc_score(np.eye(9)[tgts][:, present], probs[:, present], average="macro", multi_class="ovr"))
    except ValueError:
        auc = float("nan")
    print(f"\nAccuracy {acc:.4f} | macro-F1 {f1:.4f} | macro-AUC {auc:.4f}")

    assets = os.path.abspath(args.assets_dir); os.makedirs(assets, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(best_state, args.out)

    plt.figure(figsize=(7, 5))
    for k in ("tr_loss", "va_loss", "va_acc"):
        plt.plot(hist[k], label=k.replace("_", " "))
    plt.xlabel("epoch"); plt.legend(); plt.title(f"Training on real CPSC-2018 ({device.type})")
    plt.tight_layout(); plt.savefig(os.path.join(assets, "real_run_training_curves.png"), dpi=120)

    cm = confusion_matrix(tgts, preds, labels=list(range(9)), normalize="true")
    plt.figure(figsize=(6.5, 6))
    plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(9), CLASS_NAMES, rotation=45, ha="right"); plt.yticks(range(9), CLASS_NAMES)
    plt.title("Confusion matrix — real data (normalized)"); plt.colorbar()
    for i in range(9):
        for j in range(9):
            plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(assets, "real_run_confusion_matrix.png"), dpi=120)

    metrics = {"records": n_recs, "windows": int(len(y)), "epochs_run": len(hist["tr_loss"]),
               "accuracy": round(acc, 4), "macro_f1": round(f1, 4), "macro_auc": round(auc, 4),
               "per_class_recall": {CLASS_NAMES[c]: round(float(cm[c, c]), 3) for c in range(9)}}
    json.dump(metrics, open(os.path.join(assets, "real_run_metrics.json"), "w"), indent=2)
    print("saved figures + checkpoint + metrics.")


if __name__ == "__main__":
    main()
