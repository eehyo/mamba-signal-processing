"""Phase 2 T5 model-result figures (F7-F15) from run directories and prediction dumps.

Inputs (all optional; a figure is skipped when its inputs are missing):
  --roots    run roots; every directory with config.json + summary.json below them is read
  --preds    directory of predict_t5.py .npz files
  --signal   scan_metrics.json from t5_local_analysis.py
Every number shown comes from the validation split (12,000 images).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

import t5_plot_style as st
from data import load_fashion

MODELS = ("mamba", "cnn")


def label(scan):
    return st.SCAN_LABELS.get(scan, scan)


def color(scan):
    return st.SCAN_COLORS.get(scan, "0.3")


# ---------------------------------------------------------------- loading
def collect_runs(roots):
    """Return {(model, scan, d_conv): {seed: record}} for completed seeds."""
    runs = defaultdict(dict)
    for root in roots:
        if not root.exists():
            continue
        for cfg_path in sorted(root.glob("**/config.json")):
            run = cfg_path.parent
            cfg = json.loads(cfg_path.read_text())
            if cfg.get("dataset") != "fashion":
                continue
            key = (cfg["model"], cfg["scan"], int(cfg.get("d_conv", 4)) if cfg["model"] == "mamba" else 4)
            for seed_dir in sorted(run.glob("seed_*")):
                val, hist = seed_dir / "validation.json", seed_dir / "history.json"
                if not val.exists():  # history.json is optional (published checkpoints lack it)
                    continue
                seed = int(seed_dir.name.split("_")[1])
                if seed in runs[key]:
                    print(f"warning: duplicate {key} seed {seed} in {run}; keeping first")
                    continue
                v = json.loads(val.read_text())
                runs[key][seed] = {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
                                   "epoch": v["epoch"], "history": json.loads(hist.read_text()) if hist.exists() else None,
                                   "config": cfg, "dir": str(seed_dir)}
    return runs


def collect_preds(pred_dir):
    preds = {}
    if pred_dir and pred_dir.exists():
        for f in sorted(pred_dir.glob("*.npz")):
            z = np.load(f)
            key = (str(z["model"]), str(z["scan"]), int(z["d_conv"]) if str(z["model"]) == "mamba" else 4)
            preds.setdefault(key, {})[int(z["seed"])] = {k: z[k] for k in ("y", "pred", "logits", "val_index")}
    return preds


def scans_for(runs, model, d_conv=4, order=st.ALL_SCANS):
    return [s for s in order if runs.get((model, s, d_conv))]


def stat(runs, key, metric="macro_f1"):
    vals = np.array([r[metric] for r in runs[key].values()])
    return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0), vals


# ---------------------------------------------------------------- figures
def f7_performance(runs, fig_dir, rows):
    models = [m for m in MODELS if scans_for(runs, m)]
    if not models:
        return False
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.4), layout="constrained",
                             squeeze=False)
    for ax, m in zip(axes[0], models):
        scans = scans_for(runs, m)
        lo, hi = 1, 0
        for i, s in enumerate(scans):
            mean, std, vals = stat(runs, (m, s, 4))
            acc_mean, acc_std, _ = stat(runs, (m, s, 4), "accuracy")
            rows.append({"model": m, "scan": s, "d_conv": 4, "seeds": len(vals),
                         "macro_f1_mean": mean, "macro_f1_std": std,
                         "accuracy_mean": acc_mean, "accuracy_std": acc_std})
            ax.bar(i, mean, 0.65, color=color(s), alpha=0.85,
                   hatch="//" if s == "spiral_inward" else None, edgecolor="white")
            ax.errorbar(i, mean, yerr=std, color="k", capsize=4, lw=1.2)
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals,
                       color="k", s=14, zorder=3)
            ax.text(i, mean + std + 0.004, f"{mean:.3f}", ha="center", fontsize=8)
            lo, hi = min(lo, vals.min()), max(hi, vals.max())
        ax.set_ylim(lo - 0.04, hi + 0.025)
        ax.set_xticks(range(len(scans)), [label(s) for s in scans], rotation=20)
        ax.set(ylabel="Validation macro-F1", title=f"{st.MODEL_LABELS[m]} (bar = seed mean ± std, dots = seeds)")
    fig.suptitle("F7. Classification performance by scan order (note: y-axis range differs per model)", fontsize=13)
    st.save(fig, fig_dir / "F7_performance.png")
    return True


def f8_paired(runs, fig_dir, rows):
    models = [m for m in MODELS if runs.get((m, "raster", 4))]
    if not models:
        return False
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.2), layout="constrained",
                             squeeze=False)
    markers = ["o", "s", "^", "D", "v"]
    for ax, m in zip(axes[0], models):
        base = runs[(m, "raster", 4)]
        scans = [s for s in scans_for(runs, m) if s != "raster"]
        for i, s in enumerate(scans):
            seeds = sorted(set(base) & set(runs[(m, s, 4)]))
            d = np.array([runs[(m, s, 4)][k]["macro_f1"] - base[k]["macro_f1"] for k in seeds])
            for j, (k, v) in enumerate(zip(seeds, d)):
                ax.scatter(i + (j - 1) * 0.12, v, color=color(s), marker=markers[k % 5], s=40,
                           edgecolor="k", lw=0.5, label=f"seed {k}" if i == 0 else None)
            ax.hlines(d.mean(), i - 0.3, i + 0.3, color=color(s), lw=2.5)
            if len(d) < 3:
                same = f"only {len(d)} seed(s)"
            else:
                same = "consistent ↑" if (d > 0).all() else "consistent ↓" if (d < 0).all() else "mixed"
            ax.annotate(f"{same}\n{d.mean():+.3f}", (i, max(d.max(), 0)), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=8)
            rows.append({"model": m, "scan": s, "seeds": len(d), "mean_diff_vs_raster": d.mean(),
                         "positive_seeds": int((d > 0).sum()), "consistency": same})
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(scans)), [label(s) for s in scans], rotation=20)
        ax.margins(y=0.25)
        ax.set(ylabel="macro-F1 difference (scan − Raster, same seed)",
               title=f"{st.MODEL_LABELS[m]}: difference vs Raster")
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("F8. Seed-paired difference from Raster (horizontal line = mean)", fontsize=13)
    st.save(fig, fig_dir / "F8_paired_vs_raster.png")
    return True


def f9_curves(runs, fig_dir):
    has_hist = lambda m, s: all(r["history"] for r in runs[(m, s, 4)].values())
    models = [m for m in MODELS if any(has_hist(m, s) for s in scans_for(runs, m))]
    if not models:
        return False
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.8 * len(models)), layout="constrained",
                             squeeze=False)
    for (a1, a2), m in zip(axes, models):
        for s in [s for s in scans_for(runs, m) if has_hist(m, s)]:
            recs = list(runs[(m, s, 4)].values())
            n = min(len(r["history"]) for r in recs)
            f1 = np.array([[h["macro_f1"] for h in r["history"][:n]] for r in recs])
            loss = np.array([[h["train_loss"] for h in r["history"][:n]] for r in recs])
            ep = np.arange(1, n + 1)
            for ax, v in ((a1, f1), (a2, loss)):
                ax.plot(ep, v.mean(0), color=color(s), lw=1.5, label=label(s),
                        ls="--" if s == "spiral_inward" else "-")
                ax.fill_between(ep, v.min(0), v.max(0), color=color(s), alpha=0.15)
            for r in recs:
                a1.scatter(r["epoch"], r["macro_f1"], color=color(s), marker="*", s=50, zorder=3)
        a1.set(xlabel="epoch", ylabel="Validation macro-F1", title=f"{st.MODEL_LABELS[m]}: validation (★ = saved best)")
        missing = [label(s) for s in scans_for(runs, m) if not has_hist(m, s)]
        if missing:
            a1.text(0.98, 0.04, "no per-epoch history published for:\n" + ", ".join(missing),
                    transform=a1.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")
        a2.set(xlabel="epoch", ylabel="Training loss", title=f"{st.MODEL_LABELS[m]}: training loss")
        a1.legend(fontsize=7)
    fig.suptitle("F9. Learning curves (line = seed mean, band = seed min–max)", fontsize=13)
    st.save(fig, fig_dir / "F9_learning_curves.png")
    return True


def per_class_f1(p):
    return f1_score(p["y"], p["pred"], labels=np.arange(10), average=None, zero_division=0)


def f10_per_class(preds, fig_dir, rows):
    made = False
    for m in MODELS:
        scans = [s for s in st.ALL_SCANS if preds.get((m, s, 4))]
        if len(scans) < 2 or "raster" not in scans:
            continue
        mat = np.array([np.mean([per_class_f1(p) for p in preds[(m, s, 4)].values()], 0) for s in scans])
        for s, r in zip(scans, mat):
            rows.extend({"model": m, "scan": s, "class": c, "f1": v} for c, v in zip(st.CLASS_NAMES, r))
        delta = mat - mat[scans.index("raster")]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 0.55 * len(scans) + 2.6), layout="constrained")
        im = a1.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        lim = max(0.02, np.abs(delta).max())
        im2 = a2.imshow(delta, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for ax, M, fmt in ((a1, mat, "{:.2f}"), (a2, delta, "{:+.3f}")):
            for i in range(len(scans)):
                for j in range(10):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if (ax is a1 and M[i, j] < 0.6) else "black")
            ax.set_xticks(range(10), st.CLASS_NAMES, rotation=35, ha="right")
            ax.set_yticks(range(len(scans)), [label(s) for s in scans])
        a1.set_title("Per-class F1 (seed mean)")
        a2.set_title("Difference vs Raster (blue = better, red = worse)")
        fig.colorbar(im, ax=a1, shrink=0.8)
        fig.colorbar(im2, ax=a2, shrink=0.8)
        fig.suptitle(f"F10. {st.MODEL_LABELS[m]}: which classes are affected by scan order?", fontsize=13)
        st.save(fig, fig_dir / f"F10_per_class_f1_{m}.png")
        made = True
    return made


def confusion(p):
    cm = np.zeros((10, 10))
    np.add.at(cm, (p["y"], p["pred"]), 1)
    return cm


def f11_confusion(preds, fig_dir):
    made = False
    for m in MODELS:
        if not preds.get((m, "raster", 4)):
            continue
        other = next((s for s in ("spiral", "snake", "blockwise") if preds.get((m, s, 4))), None)
        if other is None:
            continue
        seeds = sorted(set(preds[(m, "raster", 4)]) & set(preds[(m, other, 4)]))
        cms = {}
        for s in ("raster", other):
            cm = sum(confusion(preds[(m, s, 4)][k]) for k in seeds)
            cms[s] = cm / cm.sum(1, keepdims=True)
        diff = cms[other] - cms["raster"]
        np.fill_diagonal(diff, np.nan)   # diagonal = accuracy, opposite meaning to mistakes
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), layout="constrained")
        for ax, (title, M, cmap, vmin, vmax) in zip(axes, (
                (f"{label('raster')}", cms["raster"], "Blues", 0, 1),
                (f"{label(other)}", cms[other], "Blues", 0, 1),
                (f"{label(other)} − Raster, diagonal excluded", diff, "RdBu_r", -np.nanmax(np.abs(diff)),
                 np.nanmax(np.abs(diff))))):
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
            for i in range(10):
                for j in range(10):
                    v = M[i, j]
                    if np.isfinite(v) and abs(v) >= 0.005:
                        ax.text(j, i, f"{v*100:+.1f}" if ax is axes[2] else f"{v*100:.0f}",
                                ha="center", va="center", fontsize=6.5,
                                color="white" if (ax is not axes[2] and v > 0.6) else "black")
            ax.set_xticks(range(10), st.CLASS_NAMES, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(10), st.CLASS_NAMES, fontsize=8)
            ax.set(xlabel="Predicted", ylabel="True", title=title + f" (row %, {len(seeds)} seeds pooled)")
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f"F11. {st.MODEL_LABELS[m]}: confusions (right: red = this mistake becomes more frequent)",
                     fontsize=13)
        st.save(fig, fig_dir / f"F11_confusion_{m}.png")
        made = True
    return made


def f12_disagreement(preds, fig_dir, data_root):
    made = False
    images = None
    for m in MODELS:
        scans = [s for s in st.ALL_SCANS if preds.get((m, s, 4))]
        if len(scans) < 2:
            continue
        n = len(scans)
        D = np.zeros((n, n))
        for i, a in enumerate(scans):
            for j, b in enumerate(scans):
                seeds = sorted(set(preds[(m, a, 4)]) & set(preds[(m, b, 4)]))
                D[i, j] = np.mean([(preds[(m, a, 4)][k]["pred"] != preds[(m, b, 4)][k]["pred"]).mean()
                                   for k in seeds])
        fig = plt.figure(figsize=(16, 6), layout="constrained")
        gs = fig.add_gridspec(2, 7, width_ratios=[2.4, 2.4, 1, 1, 1, 1, 1])
        ax = fig.add_subplot(gs[:, :2])
        im = ax.imshow(D * 100, cmap="Oranges")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{D[i, j]*100:.1f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(n), [label(s) for s in scans], rotation=30, ha="right")
        ax.set_yticks(range(n), [label(s) for s in scans])
        ax.set_title("Images where two scans disagree (%, same-seed mean)")
        fig.colorbar(im, ax=ax, shrink=0.7)
        other = next((s for s in ("spiral", "snake") if s in scans and s != "raster"), None)
        if other and "raster" in scans:
            if images is None:
                images, _ = load_fashion(data_root, "train")
            ra, ot = preds[(m, "raster", 4)], preds[(m, other, 4)]
            seeds = sorted(set(ra) & set(ot))
            y, vidx = ra[seeds[0]]["y"], ra[seeds[0]]["val_index"]
            r_ok = np.all([ra[k]["pred"] == y for k in seeds], 0)
            o_ok = np.all([ot[k]["pred"] == y for k in seeds], 0)
            r_wrong = np.all([ra[k]["pred"] != y for k in seeds], 0)
            o_wrong = np.all([ot[k]["pred"] != y for k in seeds], 0)
            for row, (sel, tag) in enumerate(((o_ok & r_wrong, f"only {label(other)} correct"),
                                              (r_ok & o_wrong, "only Raster correct"))):
                idx = np.flatnonzero(sel)[:5]
                for c in range(5):
                    a = fig.add_subplot(gs[row, 2 + c])
                    a.axis("off")
                    if c == 0:
                        a.text(-0.12, 0.5, f"{tag}\n{sel.sum()} images", transform=a.transAxes,
                               rotation=90, ha="center", va="center", fontsize=9, weight="bold")
                    if c >= len(idx):
                        continue
                    i = idx[c]
                    wrong = ra[seeds[0]]["pred"][i] if row == 0 else ot[seeds[0]]["pred"][i]
                    a.imshow(images[vidx[i]], cmap="gray_r")
                    a.set_title(f"true: {st.CLASS_NAMES[y[i]]}\nwrong: {st.CLASS_NAMES[wrong]}", fontsize=7)
        fig.suptitle(f"F12. {st.MODEL_LABELS[m]}: images whose prediction depends on scan order "
                     "(right: only one scan correct in every seed)", fontsize=13)
        st.save(fig, fig_dir / f"F12_disagreement_{m}.png")
        made = True
    return made


def f13_signal_vs_perf(runs, signal, fig_dir):
    if not signal:
        return False
    models = [m for m in MODELS if scans_for(runs, m)]
    if not models:
        return False
    metrics = (("jump_window_fraction_w4", "4-token windows with a spatial jump", 100, "%"),
               ("mean_abs_pixel_diff", "Mean |pixel change|", 1, ""),
               ("harmonic_energy_share", "Energy at period-14 harmonics", 100, "%"))
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4 * len(models)), layout="constrained",
                             squeeze=False)
    for r, m in enumerate(models):
        for ax, (key, name, scale, unit) in zip(axes[r], metrics):
            for s in scans_for(runs, m):
                if s not in signal:
                    continue
                mean, std, _ = stat(runs, (m, s, 4))
                x = signal[s][key] * scale
                hollow = s == "spiral_inward"
                ax.errorbar(x, mean, yerr=std, fmt="o", ms=9, capsize=3,
                            color=color(s), mfc="white" if hollow else color(s))
                ax.annotate(label(s), (x, mean), xytext=(6, 6 if not hollow else -12),
                            textcoords="offset points", fontsize=8)
            ax.set(xlabel=f"{name} {('(' + unit + ')') if unit else ''}",
                   ylabel=f"{st.MODEL_LABELS[m]} validation macro-F1")
            ax.margins(x=0.15, y=0.15)
    fig.suptitle("F13. Scan signal properties vs performance (only 5–6 points: not causal evidence)", fontsize=13)
    st.save(fig, fig_dir / "F13_signal_vs_performance.png")
    return True


def f14_ablation(runs, fig_dir, rows):
    ks = sorted({k for (m, s, k) in runs if m == "mamba" and runs[(m, s, k)]})
    if len(ks) < 2 or not all(runs.get(("mamba", "raster", k)) for k in ks):
        return False
    scans = [s for s in st.ALL_SCANS if s != "raster" and all(runs.get(("mamba", s, k)) for k in ks)]
    if not scans:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    width = 0.8 / len(ks)
    for j, k in enumerate(ks):
        base = runs[("mamba", "raster", k)]
        gaps = []
        for s in scans:
            seeds = sorted(set(base) & set(runs[("mamba", s, k)]))
            d = np.array([runs[("mamba", s, k)][q]["macro_f1"] - base[q]["macro_f1"] for q in seeds])
            gaps.append(d)
            rows.append({"d_conv": k, "scan": s, "gap_vs_raster": d.mean(), "seeds": len(d)})
        x = np.arange(len(scans)) + (j - (len(ks) - 1) / 2) * width
        bars = ax.bar(x, [g.mean() for g in gaps], width, label=f"d_conv={k}",
                      color=["#D55E00", "#999999", "#56B4E9"][j % 3])
        ax.bar_label(bars, fmt="%+.3f", fontsize=7, padding=2)
        for xi, g in zip(x, gaps):
            ax.scatter(np.full(len(g), xi), g, color="k", s=10, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(scans)), [label(s) for s in scans])
    ax.set(ylabel="macro-F1 gap G vs Raster", title="Does removing the conv window (d_conv=1) shrink the scan gap?")
    ax.legend()
    fig.suptitle("F14. Causal convolution ablation", fontsize=13)
    st.save(fig, fig_dir / "F14_dconv_ablation.png")
    return True


def f15_direction(runs, preds, fig_dir, rows):
    models = [m for m in MODELS if runs.get((m, "spiral", 4)) and runs.get((m, "spiral_inward", 4))]
    if not models:
        return False
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 4 * len(models)), layout="constrained",
                             squeeze=False, gridspec_kw={"width_ratios": [1, 2.2]})
    for (a1, a2), m in zip(axes, models):
        out, inn = runs[(m, "spiral", 4)], runs[(m, "spiral_inward", 4)]
        seeds = sorted(set(out) & set(inn))
        for k in seeds:
            a1.plot([0, 1], [out[k]["macro_f1"], inn[k]["macro_f1"]], "-o", color="0.4", lw=1)
            a1.annotate(f"seed {k}", (1, inn[k]["macro_f1"]), xytext=(5, 0),
                        textcoords="offset points", fontsize=7, va="center")
        d = np.array([inn[k]["macro_f1"] - out[k]["macro_f1"] for k in seeds])
        rows.append({"model": m, "seeds": len(d), "inward_minus_outward": d.mean(),
                     "positive_seeds": int((d > 0).sum())})
        a1.set_xticks([0, 1], ["Spiral (out)\ncentre → edge", "Spiral (in)\nedge → centre"])
        a1.set_xlim(-0.3, 1.4)
        a1.set(ylabel="Validation macro-F1", title=f"{st.MODEL_LABELS[m]}: mean difference {d.mean():+.4f}")
        po, pi = preds.get((m, "spiral", 4), {}), preds.get((m, "spiral_inward", 4), {})
        ps = sorted(set(po) & set(pi))
        if ps:
            dc = np.array([per_class_f1(pi[k]) - per_class_f1(po[k]) for k in ps])
            a2.bar(range(10), dc.mean(0), color=[st.SCAN_COLORS["spiral_inward"] if v > 0 else
                                                 st.SCAN_COLORS["spiral"] for v in dc.mean(0)])
            for k in range(len(ps)):
                a2.scatter(range(10), dc[k], color="k", s=10, zorder=3)
            a2.axhline(0, color="k", lw=0.8)
            a2.set_xticks(range(10), st.CLASS_NAMES, rotation=30, ha="right")
            a2.set(ylabel="Per-class F1 difference (in − out)",
                   title="Where does direction matter? (bar = mean, dots = seeds)")
        else:
            a2.axis("off")
            a2.text(0.5, 0.5, "no prediction files", ha="center")
    fig.suptitle("F15. Same path, reversed (identical signal properties, only processing order differs)", fontsize=13)
    st.save(fig, fig_dir / "F15_spiral_direction.png")
    return True


def write_rows(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


def main():
    root = Path(__file__).resolve().parent.parent / "phase2_t5"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", type=Path, nargs="+", default=[
        root / "outputs/t5-mamba-validation", root / "outputs/t5-mamba-validation-extra",
        root / "outputs/t5-cnn-validation", root / "outputs/t5-ablation"])
    p.add_argument("--preds", type=Path, default=root / "outputs/predictions")
    p.add_argument("--signal", type=Path, default=root / "results/train-local/scan_metrics.json")
    p.add_argument("--data_root", type=Path, default=root / "data/FashionMNIST/raw")
    p.add_argument("--figures", type=Path, default=root / "figures")
    p.add_argument("--tables", type=Path, default=root / "results/model-validation")
    args = p.parse_args()
    st.setup()
    args.tables.mkdir(parents=True, exist_ok=True)
    runs, preds = collect_runs(args.roots), collect_preds(args.preds)
    signal = json.loads(args.signal.read_text()) if args.signal.exists() else None
    print("runs:", {f"{m}/{s}/k{k}": sorted(v) for (m, s, k), v in sorted(runs.items()) if v})
    print("preds:", {f"{m}/{s}/k{k}": sorted(v) for (m, s, k), v in sorted(preds.items())})
    perf, paired, cls, abl, direction = [], [], [], [], []
    made = {
        "F7": f7_performance(runs, args.figures, perf),
        "F8": f8_paired(runs, args.figures, paired),
        "F9": f9_curves(runs, args.figures),
        "F10": f10_per_class(preds, args.figures, cls),
        "F11": f11_confusion(preds, args.figures),
        "F12": f12_disagreement(preds, args.figures, args.data_root),
        "F13": f13_signal_vs_perf(runs, signal, args.figures),
        "F14": f14_ablation(runs, args.figures, abl),
        "F15": f15_direction(runs, preds, args.figures, direction),
    }
    for name, rows in (("performance", perf), ("paired_vs_raster", paired), ("per_class_f1", cls),
                       ("dconv_ablation", abl), ("spiral_direction", direction)):
        write_rows(args.tables / f"{name}.csv", rows)
    print("figures:", {k: ("made" if v else "skipped (inputs missing)") for k, v in made.items()})


if __name__ == "__main__":
    main()
