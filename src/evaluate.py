"""
Evaluate a trained checkpoint: accuracy, macro-F1, one-vs-rest AUC, and a
normalized confusion matrix (matching the figures in the report).

Run:
    python src/evaluate.py --data_dir data/processed --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, ECGWindowDataset
from model import NUM_CLASSES, ECGArrhythmiaNet
from train import load_split


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)
        probs.append(p.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = y_prob.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(
        np.eye(NUM_CLASSES)[y_true], y_prob, average="macro", multi_class="ovr"
    )
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    return acc, f1, auc, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, X_val, y_val = load_split(args.data_dir)
    loader = DataLoader(ECGWindowDataset(X_val, y_val), batch_size=64)

    model = ECGArrhythmiaNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    y_prob, y_true = predict(model, loader, device)
    acc, f1, auc, cm = evaluate(y_true, y_prob)

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")
    print(f"Macro-AUC: {auc:.4f}")
    print("\nPer-class recall (confusion-matrix diagonal):")
    for name, r in zip(CLASS_NAMES, np.diag(cm)):
        print(f"  {name:7s}: {r:.2f}")


if __name__ == "__main__":
    main()
