"""Presentation figures from the official test evaluation (outputs/t5-test, FINAL_PROTOCOL.md).

T1  Mamba test macro-F1 per scan + same-seed difference vs Raster
T2  Mamba vs 1D CNN: same-seed difference vs Raster on the test set
T3  Mamba error analysis on the test set: Raster confusion matrix + per-class F1 change vs Raster
T4  Spiral direction: the two paths and the same-seed test macro-F1, outward vs inward
Column is included automatically once its test results exist.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import t5_plot_style as st
from t5_scan import scan_indices

T5 = Path(__file__).resolve().parent.parent / "phase2_t5"
ORDER = ["raster", "snake", "column", "random", "blockwise", "spiral"]
LABELS = {**st.SCAN_LABELS, "column": "Column"}
MC, ML = st.MODEL_COLORS, st.MODEL_LABELS
GROUP_MODEL = {"mamba": "mamba", "cnn": "cnn"}


def load(root):
    runs = defaultdict(dict)  # (group, scan) -> seed -> result
    for path in root.glob("*/*/seed_*.json"):
        r = json.loads(path.read_text())
        runs[(path.parent.parent.name, path.parent.name)][int(path.stem.split("_")[1])] = r
    return runs


def per_class_f1(cm):
    cm = np.asarray(cm, float)
    tp = np.diag(cm)
    p, r = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)


def scans(runs, group, extra=()):
    return [s for s in ORDER + list(extra) if (group, s) in runs]


def diffs(runs, group, scan, base="raster"):
    a, b = runs[(group, scan)], runs[(group, base)]
    return np.array([a[k]["macro_f1"] - b[k]["macro_f1"] for k in sorted(a) if k in b])


def consistency(d):
    return f"{(d < 0).sum()}/{len(d)} lower" if (d < 0).sum() >= (d > 0).sum() else f"{(d > 0).sum()}/{len(d)} higher"


def t1(runs, out):
    ss = scans(runs, "mamba")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    rng = np.random.default_rng(0)
    for i, s in enumerate(ss):
        v = np.array([r["macro_f1"] for r in runs[("mamba", s)].values()])
        a1.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="o", ms=9, color=MC["mamba"], capsize=4, lw=2, zorder=3)
        a1.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color="0.45", zorder=2)
        a1.text(i + 0.2, v.mean(), f"{v.mean():.3f}", ha="left", va="center", fontsize=9, fontweight="bold")
    a1.set_xlim(-0.5, len(ss) - 0.2)
    a1.set_xticks(range(len(ss)), [LABELS[s] for s in ss], rotation=20)
    a1.set(ylabel="test macro-F1", title="Mamba on the test set (mean ± std, dots = seeds)")
    a1.grid(axis="y", color="0.9")
    other = [s for s in ss if s != "raster"]
    for i, s in enumerate(other):
        d = diffs(runs, "mamba", s) * 100
        a2.bar(i, d.mean(), width=0.6, color=MC["mamba"], alpha=0.35)
        a2.scatter(np.full(len(d), i), d, s=26, color=MC["mamba"], zorder=3)
        a2.text(i, min(d.min(), 0) - 0.25, consistency(d), ha="center", va="top", fontsize=8.5)
    a2.axhline(0, color="0.3", lw=1)
    a2.set_xticks(range(len(other)), [LABELS[s] for s in other], rotation=20)
    lo = min(diffs(runs, "mamba", s).min() for s in other) * 100
    a2.set_ylim(lo - 0.8, max(0.6, max(diffs(runs, "mamba", s).max() for s in other) * 100 + 0.3))
    a2.set(ylabel="difference vs Raster (%p)", title="Same seed, minus Raster")
    a2.grid(axis="y", color="0.9")
    fig.tight_layout()
    st.save(fig, out / "T1_mamba_test.png")


def t2(runs, out):
    ss = [s for s in scans(runs, "mamba") if s != "raster" and ("cnn", s) in runs]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, s in enumerate(ss):
        for j, g in enumerate(("mamba", "cnn")):
            d = diffs(runs, g, s) * 100
            x = i + (j - 0.5) * 0.36
            ax.bar(x, d.mean(), width=0.32, color=MC[g], alpha=0.35, label=ML[g] if i == 0 else None)
            ax.scatter(np.full(len(d), x), d, s=24, color=MC[g], zorder=3)
            ax.text(x, min(d.min(), 0) - 0.2, consistency(d).replace(" ", "\n", 1), ha="center", va="top",
                    fontsize=7.5)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(range(len(ss)), [LABELS[s] for s in ss])
    lo = min(min(diffs(runs, g, s).min() for s in ss) for g in ("mamba", "cnn")) * 100
    hi = max(max(diffs(runs, g, s).max() for s in ss) for g in ("mamba", "cnn")) * 100
    ax.set_ylim(lo - 1.0, max(2.2, hi + 0.4))
    ax.set(ylabel="test macro-F1 difference vs Raster (%p)",
           title="Same seed, minus Raster: Mamba vs 1D CNN (dots = seeds)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", color="0.9")
    fig.tight_layout()
    st.save(fig, out / "T2_mamba_vs_cnn_test.png")


def t3(runs, out):
    ss = scans(runs, "mamba", ("spiral_inward",))
    cm = sum(np.asarray(r["confusion_matrix"], float) for r in runs[("mamba", "raster")].values())
    pct = cm / cm.sum(1, keepdims=True) * 100
    f1 = {s: np.mean([per_class_f1(r["confusion_matrix"]) for r in runs[("mamba", s)].values()], 0) for s in ss}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    a1.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            if pct[i, j] >= 1:
                a1.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if pct[i, j] > 60 else "0.15")
    a1.set_xticks(range(10), st.CLASS_NAMES, rotation=40, ha="right", fontsize=8)
    a1.set_yticks(range(10), st.CLASS_NAMES, fontsize=8)
    a1.set(xlabel="predicted", ylabel="true", title="Mamba, Raster: where images go (row %, 3 seeds)")
    rows = [s for s in ss if s != "raster"]
    d = np.array([f1[s] - f1["raster"] for s in rows]) * 100
    lim = np.abs(d).max()
    im = a2.imshow(d, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(10):
            a2.text(j, i, f"{d[i, j]:+.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(d[i, j]) > 0.65 * lim else "0.15")
    a2.set_xticks(range(10), st.CLASS_NAMES, rotation=40, ha="right", fontsize=8)
    a2.set_yticks(range(len(rows)), [LABELS[s] for s in rows], fontsize=9)
    a2.set_title("Per-class F1 change vs Raster (%p, blue = better, red = worse)")
    fig.colorbar(im, ax=a2, fraction=0.03, pad=0.02, label="%p")
    fig.tight_layout()
    st.save(fig, out / "T3_mamba_errors_test.png")


def t4(runs, out):
    fig = plt.figure(figsize=(11.5, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.25, 2.2])
    for k, (s, title) in enumerate((("spiral", "Spiral (out): center first"), ("spiral_inward", "Spiral (in): border first"))):
        ax = fig.add_subplot(gs[0, k])
        idx = np.asarray(scan_indices(st.SCAN_CONDITION[s]))
        r, c = np.unravel_index(idx, (14, 14))
        cols = plt.cm.viridis(np.linspace(0, 1, len(idx)))
        for t in range(len(idx) - 1):
            ax.plot(c[t:t + 2], r[t:t + 2], color=cols[t], lw=2)
        ax.scatter(c[0], r[0], s=60, color=cols[0], edgecolor="k", zorder=3)
        ax.text(c[0], r[0] - 0.9, "start", ha="center", fontsize=8)
        ax.set(xlim=(-0.7, 13.7), ylim=(13.7, -0.7), aspect="equal", xticks=[], yticks=[], title=title)
        for sp in ax.spines.values():
            sp.set_visible(False)
    sub = gs[0, 3].subgridspec(1, 2, wspace=0.45)
    for j, g in enumerate(("mamba", "cnn")):
        if (g, "spiral_inward") not in runs:
            continue
        ax = fig.add_subplot(sub[0, j])
        o, i_ = runs[(g, "spiral")], runs[(g, "spiral_inward")]
        for k in sorted(o):
            ax.plot([0, 1], [o[k]["macro_f1"], i_[k]["macro_f1"]], "-o", color=MC[g], ms=6, lw=1.6)
        d = np.array([i_[k]["macro_f1"] - o[k]["macro_f1"] for k in sorted(o)]) * 100
        better = (d > 0).sum()
        ax.set_title(f"{ML[g]}\nin minus out = {d.mean():+.1f}%p\n(in better in {better}/{len(d)} seeds)", fontsize=10)
        vals = [r["macro_f1"] for r in list(o.values()) + list(i_.values())]
        mid = (max(vals) + min(vals)) / 2
        ax.set_ylim(mid - 0.012, mid + 0.012)  # same 2.4%p window for both models
        ax.set_xticks([0, 1], ["out", "in"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("test macro-F1")
        ax.grid(axis="y", color="0.9")
    fig.tight_layout()
    st.save(fig, out / "T4_spiral_direction_test.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=T5 / "outputs/t5-test")
    p.add_argument("--out", type=Path, default=T5 / "figures/test")
    args = p.parse_args()
    st.setup()
    runs = load(args.runs)
    t1(runs, args.out)
    t2(runs, args.out)
    t3(runs, args.out)
    t4(runs, args.out)
    print("wrote", sorted(x.name for x in args.out.glob("*.png")))


if __name__ == "__main__":
    main()
