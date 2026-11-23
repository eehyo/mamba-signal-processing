"""F7b: does the 1D CNN baseline change if it trains longer?

Compares the protocol run (20 epochs) with a supplementary run (up to 60 epochs,
stop after 5 epochs without a validation macro-F1 gain). Both keep the
best-validation epoch, so both report validation scores (no test data).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import t5_plot_style as style

HERE = Path(__file__).resolve().parent
T5 = HERE.parent / "phase2_t5"


def load(root):
    """scan -> {seed: (best macro_f1, best epoch, epochs run, val F1 curve)}"""
    runs = {}
    for scan in style.ALL_SCANS:
        for seed_dir in sorted((root / scan).glob("seed_*")):
            val = seed_dir / "validation.json"
            if not val.exists():
                continue
            v = json.loads(val.read_text())
            hist = json.loads((seed_dir / "history.json").read_text())
            seed = int(seed_dir.name.split("_")[1])
            runs.setdefault(scan, {})[seed] = (v["macro_f1"], v["epoch"], len(hist),
                                               [h["macro_f1"] for h in hist])
    return runs


def paired_signs(runs, scan):
    """Seed-wise sign of (scan - raster): 'n+/n' and mean difference."""
    common = sorted(set(runs.get(scan, {})) & set(runs.get("raster", {})))
    d = np.array([runs[scan][k][0] - runs["raster"][k][0] for k in common])
    return f"{int((d > 0).sum())}/{len(d)}", float(d.mean()) if len(d) else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--short", type=Path, default=T5 / "outputs/t5-cnn-validation")
    p.add_argument("--long", type=Path, default=T5 / "outputs/t5-cnn-validation-e60")
    p.add_argument("--figures", type=Path, default=T5 / "figures")
    p.add_argument("--tables", type=Path, default=T5 / "results/model-validation")
    args = p.parse_args()
    style.setup()
    import matplotlib.pyplot as plt

    short, long_ = load(args.short), load(args.long)
    scans = [s for s in style.ALL_SCANS if s in short and s in long_]

    rows = []
    for s in scans:
        a = np.array([v[0] for v in short[s].values()])
        b = np.array([v[0] for v in long_[s].values()])
        sign20, d20 = paired_signs(short, s)
        sign60, d60 = paired_signs(long_, s)
        rows.append({
            "scan": s,
            "f1_20ep_mean": a.mean(), "f1_20ep_std": a.std(ddof=1) if len(a) > 1 else float("nan"),
            "f1_60ep_mean": b.mean(), "f1_60ep_std": b.std(ddof=1) if len(b) > 1 else float("nan"),
            "gain": b.mean() - a.mean(),
            "vs_raster_20ep": d20, "vs_raster_20ep_positive": sign20,
            "vs_raster_60ep": d60, "vs_raster_60ep_positive": sign60,
            "best_epochs_60ep": " ".join(str(long_[s][k][1]) for k in sorted(long_[s])),
            "epochs_run_60ep": " ".join(str(long_[s][k][2]) for k in sorted(long_[s])),
        })
    args.tables.mkdir(parents=True, exist_ok=True)
    with open(args.tables / "cnn_epochs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(scans))
    for j, (runs, label, shift, alpha) in enumerate([(short, "20 epochs (protocol)", -0.18, 0.45),
                                                     (long_, "≤60 epochs, early stop (patience 5)", 0.18, 1.0)]):
        for i, s in enumerate(scans):
            vals = np.array([runs[s][k][0] for k in sorted(runs[s])])
            c = style.SCAN_COLORS[s]
            ax1.bar(i + shift, vals.mean(), width=0.34, color=c, alpha=alpha,
                    edgecolor="k", lw=0.6, hatch=None if j else "//",
                    label=label if i == 0 else None)
            ax1.scatter(np.full(len(vals), i + shift), vals, s=14, color="k", zorder=3)
    ax1.set_xticks(x, [style.SCAN_LABELS[s] for s in scans], rotation=20)
    lo = min(min(v[0] for v in r[s].values()) for r in (short, long_) for s in scans)
    hi = max(max(v[0] for v in r[s].values()) for r in (short, long_) for s in scans)
    ax1.set_ylim(lo - 0.02, hi + 0.03)
    ax1.set(ylabel="validation macro-F1 (best epoch)", title="1D CNN: 20 epochs vs longer training")
    ax1.legend(fontsize=8, loc="upper left")

    for s in scans:
        for k in sorted(long_[s]):
            f1, best, n, curve = long_[s][k]
            ax2.plot(np.arange(1, n + 1), curve, color=style.SCAN_COLORS[s], lw=1, alpha=0.8,
                     label=style.SCAN_LABELS[s] if k == min(long_[s]) else None)
            ax2.scatter([best], [f1], color=style.SCAN_COLORS[s], s=18, zorder=3, edgecolor="k", lw=0.4)
    ax2.axvline(20, color="k", ls=":", lw=1)
    ax2.text(20.5, ax2.get_ylim()[0] + 0.01, "protocol stops at 20", fontsize=8, va="bottom")
    ax2.set(xlabel="epoch", ylabel="validation macro-F1",
            title="Learning curves (≤60 epochs; dot = best epoch, line ends = early stop)")
    ax2.legend(fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("F7b. Supplementary: the CNN baseline was still improving at epoch 20", y=1.02)
    style.save(fig, args.figures / "F7b_cnn_epochs.png")
    print(json.dumps({r["scan"]: [round(r["f1_20ep_mean"], 4), round(r["f1_60ep_mean"], 4),
                                  r["vs_raster_60ep_positive"]] for r in rows}))


if __name__ == "__main__":
    main()
