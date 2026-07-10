"""
CNN + Bidirectional GRU + Attention model for 12-lead ECG arrhythmia classification.

Architecture (as described in the report):
    Input (B, 12, 5000)  -- 12 leads, 10 s @ 500 Hz
      -> 5x [Conv1d -> ReLU -> MaxPool(2) -> Dropout]   channels 32,64,128,256,512
      -> Bidirectional GRU (100 units per direction -> 200-dim per timestep)
      -> Additive/dot-product attention over time -> 200-dim context vector
      -> Linear(200 -> 9) + softmax

Total trainable parameters: 934,993  (verified in __main__).

NOTE FOR PORTFOLIO: this file is a faithful re-implementation of the architecture
documented in the report. Verify it against your actual training script before
presenting it as the exact code that produced the reported numbers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9
NUM_LEADS = 12
INPUT_LEN = 5000  # 10 s @ 500 Hz


class ConvBlock(nn.Module):
    """Conv1d -> ReLU -> MaxPool(stride 2) -> Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention pooling over time with a learned query vector.

    Given H of shape (B, T, D):
        scores = (tanh(H W) . q) / sqrt(D)
        alpha  = softmax(scores)          # (B, T)
        context = sum_t alpha_t * H_t     # (B, D)
    """

    def __init__(self, dim: int = 200):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.query = nn.Parameter(torch.randn(dim))
        self.scale = dim ** 0.5

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.tanh(self.proj(h))                 # (B, T, D)
        scores = u @ self.query / self.scale         # (B, T)
        alpha = torch.softmax(scores, dim=1)         # (B, T)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, D)
        return context, alpha


class ECGArrhythmiaNet(nn.Module):
    def __init__(
        self,
        num_leads: int = NUM_LEADS,
        num_classes: int = NUM_CLASSES,
        conv_channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        gru_hidden: int = 100,
        dropout: float = 0.3,
    ):
        super().__init__()
        blocks = []
        in_ch = num_leads
        for out_ch in conv_channels:
            blocks.append(ConvBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)

        self.gru = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = Attention(dim=2 * gru_hidden)
        self.classifier = nn.Linear(2 * gru_hidden, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B, 12, 5000)
        x = self.conv(x)               # (B, 512, T')
        x = x.transpose(1, 2)          # (B, T', 512)
        h, _ = self.gru(x)             # (B, T', 200)
        context, alpha = self.attention(h)
        logits = self.classifier(context)
        if return_attention:
            return logits, alpha
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ECGArrhythmiaNet()
    n = count_parameters(model)
    print(f"Trainable parameters: {n:,}")
    assert n == 934_993, f"expected 934,993 params, got {n:,}"
    dummy = torch.randn(2, NUM_LEADS, INPUT_LEN)
    out = model(dummy)
    print("Output shape:", tuple(out.shape))  # (2, 9)
    print("OK")
