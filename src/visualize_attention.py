"""
Visualize *where* the attention layer looks on real CPSC-2018 ECG.

The model ends in an attention-pooling layer over time (see `model.Attention`).
Those attention weights are a built-in interpretability signal: they say which
moments of the 10-second window the classifier relied on. This script loads the
trained checkpoint, runs the model with `return_attention=True` on a few
correctly-classified real records (one per arrhythmia), upsamples the attention
back to the original time axis, and overlays it on the lead-II waveform.

Output: assets/attention_heatmap.png

Run:
    python src/visualize_attention.py \
        --data_dir "/path/to/CPSC2018" \
        --checkpoint checkpoints/real_model.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from dataset import CLASS_NAMES
from model import ECGArrhythmiaNet
from preprocessing import preprocess_recording
from train_raw_cpsc import load_labels, load_signal

LEAD_II = 1  # standard rhythm lead in the 12-lead order [I, II, III, aVR, aVL, aVF, V1..V6]
FS = 500
WINDOW = 5000

# Clinical one-liners: what a cardiologist would look for -> a sanity check on the attention.
CLINICAL_HINT = {
    "Normal": "sinus rhythm, regular QRS",
    "AF":     "irregular rhythm, no P waves",
    "I-AVB":  "prolonged PR interval",
    "LBBB":   "wide QRS, broad R waves",
    "RBBB":   "wide QRS, RSR' pattern",
    "PAC":    "premature atrial beat",
    "PVC":    "premature wide ectopic beat",
    "STD":    "ST-segment depression",
    "STE":    "ST-segment elevation",
}

# Which arrhythmias tell the clearest interpretability story, in order of preference.
PREFERRED = ["Normal", "AF", "RBBB", "PVC", "I-AVB", "LBBB", "STD", "PAC", "STE"]


def upsample(alpha: np.ndarray, length: int = WINDOW) -> np.ndarray:
    """Stretch the (T',) attention vector back onto the original sample axis."""
    src = np.linspace(0.0, 1.0, len(alpha))
    dst = np.linspace(0.0, 1.0, length)
    a = np.interp(dst, src, alpha)
    rng = a.max() - a.min()
    return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)  # per-window contrast stretch


def find_examples(model, ref, mats, order, device, n_wanted, conf_thr, scan_cap, seed):
    """Scan records for one confident, correctly-classified example per class.

    Keep going until the top-`n_wanted` preferred classes are all filled (so the
    figure gets the clearest clinical story, not just the first classes seen),
    recording a fallback example for every other class encountered meanwhile.
    """
    recs = sorted(mats)
    np.random.default_rng(seed).shuffle(recs)
    target = set(order[:n_wanted])
    picked: dict[str, dict] = {}
    scanned = 0
    for r in recs:
        if target.issubset(picked) or scanned >= scan_cap:
            break
        name = CLASS_NAMES[ref[r]]
        if name not in order or name in picked:
            continue
        try:
            sig = load_signal(mats[r])
        except Exception:
            continue
        if sig.shape[0] != 12:
            sig = sig.T
        windows = preprocess_recording(sig)
        if len(windows) == 0:
            continue
        scanned += 1
        x = torch.as_tensor(windows[:1], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, alpha = model(x, return_attention=True)
            prob = torch.softmax(logits, 1)[0].cpu().numpy()
        pred = int(prob.argmax())
        if pred == ref[r] and prob[pred] >= conf_thr:
            picked[name] = {
                "record": r,
                "signal": windows[0],
                "alpha": alpha[0].cpu().numpy(),
                "conf": float(prob[pred]),
            }
    return picked


def make_figure(examples, order, out_path):
    heat = LinearSegmentedColormap.from_list(  # white -> amber -> deep red
        "attn", ["#ffffff", "#ffe08a", "#f08c2e", "#c92a2a"])
    names = [n for n in order if n in examples][:4]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    fig.patch.set_facecolor("white")
    t = np.arange(WINDOW) / FS

    for ax, name in zip(axes.flat, names):
        ex = examples[name]
        sig = ex["signal"][LEAD_II]
        w = upsample(ex["alpha"])
        ax.imshow(w[None, :], aspect="auto", extent=[0, WINDOW / FS, -0.06, 1.06],
                  cmap=heat, vmin=0, vmax=1, alpha=0.95, origin="lower")
        ax.plot(t, sig, color="#1b2733", lw=0.9)
        ax.set_xlim(0, WINDOW / FS)
        ax.set_ylim(-0.06, 1.06)
        ax.set_yticks([])
        ax.set_title(f"{name}   ·   predicted ✓  ({ex['conf']:.0%} conf)",
                     fontsize=12.5, fontweight="bold", color="#1b2733", pad=6)
        ax.text(0.012, 0.955, CLINICAL_HINT[name], transform=ax.transAxes,
                fontsize=9.2, style="italic", color="#5c6773", va="top",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#d7dce1", alpha=0.85))
        ax.text(0.988, 0.05, ex["record"], transform=ax.transAxes, fontsize=8,
                color="#9aa5b1", ha="right", va="bottom")
        for s in ax.spines.values():
            s.set_color("#d7dce1")
        ax.tick_params(colors="#5c6773", labelsize=9)

    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=10, color="#5c6773")
    # blank any unused panels
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")

    fig.subplots_adjust(left=0.03, right=0.98, top=0.855, bottom=0.155, hspace=0.34, wspace=0.06)

    fig.suptitle("Where the model looks — attention over lead II (real CPSC-2018 records)",
                 fontsize=15.5, fontweight="bold", color="#12181f", y=0.975)
    fig.text(0.5, 0.915,
             "Warm shading = higher attention. The network keys on QRS complexes and "
             "abnormal beats rather than the flat baseline between them.",
             ha="center", fontsize=10.3, color="#5c6773")

    cax = fig.add_axes([0.34, 0.055, 0.32, 0.019])  # dedicated strip below the time-axis labels
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=heat),
                        cax=cax, orientation="horizontal")
    cbar.set_label("attention weight (low → high)", fontsize=9.5, color="#5c6773")
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.tick_params(labelsize=8, colors="#5c6773")
    cbar.outline.set_edgecolor("#d7dce1")

    fig.savefig(out_path, dpi=140, facecolor="white")
    print("saved", out_path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="CPSC-2018 folder (.mat + REFERENCE.csv)")
    ap.add_argument("--checkpoint", default=os.path.join(here, "..", "checkpoints", "real_model.pt"))
    ap.add_argument("--out", default=os.path.join(here, "..", "assets", "attention_heatmap.png"))
    ap.add_argument("--n", type=int, default=4, help="number of panels / classes")
    ap.add_argument("--conf", type=float, default=0.80, help="min confidence for a clean example")
    ap.add_argument("--scan_cap", type=int, default=1200, help="max records to preprocess while searching")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model = ECGArrhythmiaNet().to(device).eval()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print("loaded checkpoint:", args.checkpoint, "| device:", device)

    ref = load_labels(args.data_dir)
    mats = {}
    for p in glob.glob(os.path.join(args.data_dir, "**", "*.mat"), recursive=True):
        if "CPSC0236" in p:  # champion's pretrained bundle, not our data
            continue
        b = os.path.basename(p)[:-4]
        if b in ref:
            mats.setdefault(b, p)
    print(f"labeled records available: {len(mats)}")

    examples = find_examples(model, ref, mats, PREFERRED, device,
                             n_wanted=args.n, conf_thr=args.conf,
                             scan_cap=args.scan_cap, seed=args.seed)
    print("picked:", {k: v["record"] for k, v in examples.items()})
    if len(examples) < 2:
        raise SystemExit("Not enough confident examples found; lower --conf or raise --scan_cap.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    make_figure(examples, PREFERRED, os.path.abspath(args.out))


if __name__ == "__main__":
    main()
