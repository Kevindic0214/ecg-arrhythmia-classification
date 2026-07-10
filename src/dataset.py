"""
Dataset utilities for CPSC-2018 12-lead ECG.

The CPSC-2018 REFERENCE.csv maps each recording to up to three labels (1-9):
    1 Normal | 2 AF | 3 I-AVB | 4 LBBB | 5 RBBB | 6 PAC | 7 PVC | 8 STD | 9 STE
Here we use the first label as the primary target (single-label classification).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

# Report ordering of the nine classes.
CLASS_NAMES = ["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
LABEL_TO_IDX = {i + 1: i for i in range(9)}  # CPSC labels are 1-indexed


class ECGWindowDataset(Dataset):
    """Wraps preprocessed windows.

    Args:
        signals: float array (N, 12, 5000), already preprocessed.
        labels:  int array (N,), class indices in [0, 8].
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        assert signals.ndim == 3 and signals.shape[1] == 12
        self.signals = torch.as_tensor(signals, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.signals[idx], self.labels[idx]


def class_weights(labels: np.ndarray, num_classes: int = 9) -> torch.Tensor:
    """Inverse-frequency weights for cost-sensitive learning on the imbalanced set."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.as_tensor(w, dtype=torch.float32)
