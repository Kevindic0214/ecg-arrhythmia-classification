"""
Signal preprocessing for CPSC-2018 12-lead ECG.

Pipeline (as described in the report):
    1. Butterworth high-pass filter, 1 Hz cutoff  -> remove baseline wander
    2. Wavelet denoising (db4, level 1), zero the highest-frequency detail
    3. Min-max normalization to [0, 1] per lead (zero-range signals left unchanged)
    4. Fixed-length windowing: 10 s = 5000 samples @ 500 Hz

Run:
    python src/preprocessing.py --data_dir /path/to/cpsc2018 --out_dir data/processed
"""
from __future__ import annotations

import argparse
import numpy as np
import pywt
from scipy.signal import butter, filtfilt

FS = 500           # sampling frequency (Hz)
WINDOW_SEC = 10
WINDOW = WINDOW_SEC * FS  # 5000 samples


def highpass_filter(sig: np.ndarray, cutoff: float = 1.0, fs: int = FS, order: int = 4) -> np.ndarray:
    """Butterworth high-pass to remove low-frequency baseline wander."""
    b, a = butter(order, cutoff / (fs / 2.0), btype="high")
    return filtfilt(b, a, sig, axis=-1)


def wavelet_denoise(sig: np.ndarray, wavelet: str = "db4", level: int = 1) -> np.ndarray:
    """Wavelet denoise: zero the highest-frequency detail coefficients on reconstruction."""
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    coeffs[-1] = np.zeros_like(coeffs[-1])  # drop highest-frequency band
    rec = pywt.waverec(coeffs, wavelet)
    return rec[..., : sig.shape[-1]]


def minmax_normalize(sig: np.ndarray) -> np.ndarray:
    """Per-lead min-max to [0, 1]; leave zero-range leads untouched (avoid /0)."""
    lo = sig.min(axis=-1, keepdims=True)
    hi = sig.max(axis=-1, keepdims=True)
    rng = hi - lo
    out = np.where(rng > 0, (sig - lo) / np.where(rng > 0, rng, 1.0), sig)
    return out


def to_windows(sig: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Pad/crop a (leads, T) recording to non-overlapping fixed windows -> (n, leads, window)."""
    n_leads, t = sig.shape
    if t < window:
        pad = np.zeros((n_leads, window - t), dtype=sig.dtype)
        sig = np.concatenate([sig, pad], axis=1)
        t = window
    n = t // window
    sig = sig[:, : n * window]
    return sig.reshape(n_leads, n, window).transpose(1, 0, 2)  # (n, leads, window)


def preprocess_recording(sig: np.ndarray) -> np.ndarray:
    """Full pipeline for a single (leads, T) recording -> (n_windows, leads, WINDOW)."""
    sig = highpass_filter(sig)
    sig = wavelet_denoise(sig)
    sig = minmax_normalize(sig)
    return to_windows(sig)


def main():
    ap = argparse.ArgumentParser(description="Preprocess CPSC-2018 ECG recordings.")
    ap.add_argument("--data_dir", required=True, help="Directory of raw CPSC-2018 recordings")
    ap.add_argument("--out_dir", required=True, help="Output directory for processed windows")
    args = ap.parse_args()
    # TODO: iterate over your recording files (.mat/.hdf5), read the 12-lead array,
    #       call preprocess_recording(), and save windows + labels to --out_dir.
    print("Preprocessing config:", dict(fs=FS, window=WINDOW, wavelet="db4", hp_cutoff=1.0))
    print("Implement the file-reading loop for your dataset layout in main().")


if __name__ == "__main__":
    main()
