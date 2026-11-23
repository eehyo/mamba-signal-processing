"""Phase 2 T5 model-free analysis that complements analyze_t5.py.

Adds what the original analysis did not cover:
  F1  scan paths including the inward spiral control
  F2  local structure inside the Mamba conv window (4 tokens; 7 for two stacked layers)
  F3  Raster row-boundary vs interior pixel changes (H1)
  F4  strength of the period-14 comb (prominence, energy share; boxcar vs Hann)
  F5  per-class spectra and continuity
  F6  Spiral vs Snake: visit order, centre distance, early/mid/late statistics
Runs on the official training split only (60,000 images).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import periodogram

from data import load_fashion
from t5_preprocess import downsample
from t5_scan import SIZE, apply_scan, scan_indices
import t5_plot_style as st
from analyze_t5 import write_csv

N = SIZE * SIZE
WINDOWS = (4, 7)          # d_conv=4; two stacked layers see 1 + 3 + 3 = 7 tokens
PERIOD = SIZE             # row length
HARMONIC_BINS = [N // PERIOD * h for h in range(1, PERIOD // 2 + 1)]  # 14, 28, ..., 98
CENTRE = (SIZE - 1) / 2   # 6.5


def coords(indices):
    return np.column_stack(np.divmod(indices, SIZE)).astype(float)


def window_stats(xy, w):
    """Per window of w consecutive tokens: contains a non-adjacent move? spatial extent."""
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    jumps = sliding_window_view(step > 1 + 1e-9, w - 1).any(axis=1)
    win = sliding_window_view(xy, w, axis=0)            # [n_win, 2, w]
    diff = win[:, :, :, None] - win[:, :, None, :]
    extent = np.sqrt((diff ** 2).sum(1)).max(axis=(1, 2))
    path = sliding_window_view(step, w - 1).sum(axis=1)
    return jumps, extent, path


def causal_extent_map(xy, w=4):
    """For each pixel: max 2D distance to the w-1 tokens processed just before it."""
    out = np.zeros((SIZE, SIZE))
    for t in range(1, N):
        prev = xy[max(0, t - w + 1):t]
        r, c = xy[t].astype(int)
        out[r, c] = np.linalg.norm(prev - xy[t], axis=1).max()
    return out


def mean_psd(seq, window):
    f, p = periodogram(seq, fs=1, window=window, detrend="constant", scaling="density")
    return f, p.mean(axis=0, dtype=np.float64)


def comb_metrics(p):
    """Energy share at k/14 harmonics and prominence of the 1/14 fundamental."""
    total = p[1:].sum()
    share = p[HARMONIC_BINS].sum() / total
    b = HARMONIC_BINS[0]
    neighbours = np.r_[p[b - 6:b - 1], p[b + 2:b + 7]]
    return float(share), float(p[b] / np.median(neighbours))


def main(args):
    st.setup()
    out, fig_dir = args.output, args.figures
    out.mkdir(parents=True, exist_ok=True)
    raw, labels = load_fashion(args.data, "train")
    images = downsample(raw)
    print(f"images {images.shape}, labels {np.bincount(labels).tolist()}", flush=True)

    scans = st.ALL_SCANS
    order = {s: scan_indices(st.SCAN_CONDITION[s]) for s in scans}
    xy = {s: coords(order[s]) for s in scans}
    seqs = {s: apply_scan(images, order[s]) for s in scans}
    metrics = {s: {} for s in scans}
    checks = {}

    # ---------- F1: scan paths ----------
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), layout="constrained")
    for ax, s in zip(axes.flat, scans):
        r, c = xy[s][:, 0], xy[s][:, 1]
        ax.plot(c, r, color="0.75", lw=0.8, zorder=1)
        sc = ax.scatter(c, r, c=np.arange(N), cmap="viridis", s=14, zorder=2)
        ax.scatter(c[0], r[0], marker="*", s=180, color="red", zorder=3, label="start")
        ax.scatter(c[-1], r[-1], marker="X", s=90, color="black", zorder=3, label="end")
        ax.set(title=st.SCAN_LABELS[s], aspect="equal", xticks=[], yticks=[])
        ax.invert_yaxis()
    axes.flat[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.colorbar(sc, ax=axes, shrink=0.6, label="Visit order (0 → 195)")
    fig.suptitle("F1. Six ways to unroll a 14×14 image into 196 tokens", fontsize=13)
    st.save(fig, fig_dir / "F1_scan_paths.png")

    # ---------- F2: local structure inside the conv window ----------
    rows = []
    for s in scans:
        step = np.linalg.norm(np.diff(xy[s], axis=0), axis=1)
        metrics[s]["mean_coordinate_distance"] = float(step.mean())
        metrics[s]["nonadjacent_moves"] = int((step > 1 + 1e-9).sum())
        for w in WINDOWS:
            jumps, extent, path = window_stats(xy[s], w)
            rows.append({"scan": s, "window": w, "windows": len(jumps),
                         "nonadjacent_moves": int((step > 1 + 1e-9).sum()),
                         "windows_with_jump": int(jumps.sum()),
                         "jump_window_fraction": float(jumps.mean()),
                         "mean_extent": float(extent.mean()), "max_extent": float(extent.max()),
                         "mean_path_length": float(path.mean())})
            metrics[s][f"jump_window_fraction_w{w}"] = float(jumps.mean())
            metrics[s][f"mean_window_extent_w{w}"] = float(extent.mean())
    write_csv(out / "window_structure.csv", rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(len(scans))
    for k, w in enumerate(WINDOWS):
        vals = [next(r["jump_window_fraction"] for r in rows if r["scan"] == s and r["window"] == w)
                for s in scans]
        ext = [next(r["mean_extent"] for r in rows if r["scan"] == s and r["window"] == w)
               for s in scans]
        hatch = None if w == 4 else "//"
        b = a1.bar(x + (k - 0.5) * 0.38, np.array(vals) * 100, 0.38,
                   color=[st.SCAN_COLORS[s] for s in scans], hatch=hatch, edgecolor="white",
                   label=f"{w} tokens" + (" (1 layer)" if w == 4 else " (2 layers)"))
        a1.bar_label(b, fmt="%.0f", fontsize=8, padding=1)
        a2.bar(x + (k - 0.5) * 0.38, ext, 0.38, color=[st.SCAN_COLORS[s] for s in scans],
               hatch=hatch, edgecolor="white", label=f"{w}-token window")
    for a in (a1, a2):
        a.set_xticks(x, [st.SCAN_LABELS[s] for s in scans], rotation=20)
        a.legend(fontsize=8, loc="center right")
    a1.set(ylabel="Windows containing a spatial jump (dist > 1, %)", ylim=(0, 112),
           title="Does the Mamba conv window contain a spatial jump?")
    a2.set(ylabel="Max 2D distance within window (pixels)",
           title="How spread out is the window in the image?")
    fig.suptitle("F2a. Local window structure: differences hidden by mean step distance", fontsize=13)
    st.save(fig, fig_dir / "F2a_window_jumps.png")

    fig, axes = plt.subplots(1, len(scans), figsize=(15, 3.2), layout="constrained")
    vmax = max(causal_extent_map(xy[s]).max() for s in scans if s != "random")
    for ax, s in zip(axes, scans):
        m = causal_extent_map(xy[s])
        im = ax.imshow(m, cmap="magma_r", vmin=0, vmax=vmax)
        ax.set(title=st.SCAN_LABELS[s], xticks=[], yticks=[])
        if s == "random":
            ax.set_title(f"Random\n(max {m.max():.1f}, beyond colour scale)", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Max distance to previous 3 tokens (pixels)")
    fig.suptitle("F2b. How far the conv window (previous 3 tokens) reaches when each pixel is processed (lighter = more local)",
                 fontsize=11)
    st.save(fig, fig_dir / "F2b_causal_window_map.png")

    sample = int(np.flatnonzero(labels == 4)[0])  # first Coat
    fig, axes = plt.subplots(1, len(scans), figsize=(15, 3.3), layout="constrained")
    for ax, s in zip(axes, scans):
        _, extent, _ = window_stats(xy[s], 4)
        j = int(np.argmax(extent))
        ax.imshow(images[sample], cmap="gray_r", vmin=0, vmax=1)
        ax.plot(xy[s][:, 1], xy[s][:, 0], color="0.6", lw=0.4, alpha=0.6)
        seg = xy[s][j:j + 4]
        ax.plot(seg[:, 1], seg[:, 0], "-o", color=st.SCAN_COLORS[s], lw=2.2, ms=6)
        for n, (r, c) in enumerate(seg):
            ax.annotate(str(n + 1), (c, r), color="white", ha="center", va="center", fontsize=7,
                        weight="bold")
        ax.set(title=f"{st.SCAN_LABELS[s]}\nwidest 4-token window: {extent[j]:.1f}px",
               xticks=[], yticks=[])
    fig.suptitle("F2c. Worst-case 4-token window on a real image (numbers = processing order)", fontsize=12)
    st.save(fig, fig_dir / "F2c_window_examples.png")

    # ---------- F3: Raster row boundaries vs interior ----------
    diffs = {s: np.abs(np.diff(seqs[s], axis=1)) for s in ("raster", "snake")}
    boundary = np.zeros(N - 1, bool)
    boundary[PERIOD - 1::PERIOD] = True          # transitions 13, 27, ..., 181
    assert boundary.sum() == SIZE - 1
    f3 = []
    for s, d in diffs.items():
        b, i = d[:, boundary].mean(1), d[:, ~boundary].mean(1)
        f3.append({"scan": s, "boundary_mean": float(b.mean()), "interior_mean": float(i.mean()),
                   "ratio": float(b.mean() / i.mean()),
                   "images_boundary_gt_interior": float((b > i).mean())})
        metrics[s]["row_boundary_ratio"] = f3[-1]["ratio"]
    write_csv(out / "row_boundary.csv", f3)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained",
                                 gridspec_kw={"width_ratios": [2.2, 1]})
    for s, d in diffs.items():
        a1.plot(np.arange(N - 1), d.mean(0), color=st.SCAN_COLORS[s], lw=1.3,
                label=st.SCAN_LABELS[s])
    for t in np.flatnonzero(boundary):
        a1.axvline(t, color="0.8", lw=0.6, zorder=0)
    a1.set(xlabel="Transition position (t → t+1)", ylabel="Mean |pixel change|",
           title="Changes are smaller at row boundaries (grey lines): both edges are mostly background")
    a1.legend()
    data, ticks = [], []
    for s, d in diffs.items():
        data += [d[:, boundary].mean(1), d[:, ~boundary].mean(1)]
        ticks += [f"{st.SCAN_LABELS[s]}\nrow boundary", f"{st.SCAN_LABELS[s]}\nwithin row"]
    parts = a2.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, s in zip(parts["boxes"], ["raster", "raster", "snake", "snake"]):
        patch.set_facecolor(st.SCAN_COLORS[s])
        patch.set_alpha(0.55)
    a2.set_xticks(range(1, 5), ticks, fontsize=8)
    a2.set(ylabel="Per-image mean |pixel change|", title="Distribution over 60,000 images")
    fig.suptitle("F3. H1 test: row-wrap jumps do not create pixel discontinuities (82% of edge pixels are near-black)",
                 fontsize=13)
    st.save(fig, fig_dir / "F3_row_boundary.png")

    # ---------- F4: period-14 comb strength ----------
    spectra, f4 = {}, []
    for s in scans:
        for window in ("boxcar", "hann"):
            f, p = mean_psd(seqs[s], window)
            spectra[s, window] = p
            share, prom = comb_metrics(p)
            f4.append({"scan": s, "window": window, "harmonic_energy_share": share,
                       "fundamental_prominence": prom,
                       "top_bin": int(np.argmax(p[1:]) + 1),
                       "top_period": float(N / (np.argmax(p[1:]) + 1))})
            if window == "boxcar":
                metrics[s]["harmonic_energy_share"] = share
                metrics[s]["fundamental_prominence"] = prom
    freq = f
    write_csv(out / "comb_strength.csv", f4)
    write_csv(out / "pixel_psd_hann_vs_boxcar.csv",
              [{"scan": s, "cycles_per_index": float(fr), "boxcar": float(spectra[s, "boxcar"][k]),
                "hann": float(spectra[s, "hann"][k])}
               for s in scans for k, fr in enumerate(freq)])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    for s in st.MAIN_SCANS:
        a1.semilogy(freq[1:], spectra[s, "boxcar"][1:], color=st.SCAN_COLORS[s], lw=1.2,
                    label=st.SCAN_LABELS[s])
    for b in HARMONIC_BINS:
        a1.axvline(freq[b], color="0.7", ls="--", lw=0.6, zorder=0)
    a1.set(xlabel="Frequency (cycles / token)", ylabel="Mean PSD (log)",
           title="Dashed = period-14 frequency (1/14) and harmonics")
    a1.legend(fontsize=8)
    x = np.arange(len(st.MAIN_SCANS))
    for k, window in enumerate(("boxcar", "hann")):
        vals = [next(r["harmonic_energy_share"] for r in f4 if r["scan"] == s and r["window"] == window)
                for s in st.MAIN_SCANS]
        b = a2.bar(x + (k - 0.5) * 0.38, np.array(vals) * 100, 0.38,
                   color=[st.SCAN_COLORS[s] for s in st.MAIN_SCANS],
                   hatch=None if window == "boxcar" else "//", edgecolor="white",
                   label=f"{window} window")
        a2.bar_label(b, fmt="%.0f", fontsize=7, padding=1)
    a2.axhline(len(HARMONIC_BINS) / (N // 2) * 100, color="k", lw=0.8, ls=":",
               label="expected without periodicity")
    a2.set_xticks(x, [st.SCAN_LABELS[s] for s in st.MAIN_SCANS], rotation=20)
    a2.set(ylabel="Energy at period-14 harmonics (%)", title="Strength of periodicity")
    a2.legend(fontsize=7)
    fig.suptitle("F4. Periodicity measured by strength of the period-14 comb, not peak location", fontsize=13)
    st.save(fig, fig_dir / "F4_periodicity_strength.png")

    # ---------- F5: per-class spectra and continuity ----------
    f5, cls_psd = [], {}
    for s in st.MAIN_SCANS:
        d = np.abs(np.diff(seqs[s], axis=1)).mean(1)
        for c in range(10):
            sel = labels == c
            _, p = mean_psd(seqs[s][sel], "boxcar")
            cls_psd[s, c] = p
            share, prom = comb_metrics(p)
            f5.append({"scan": s, "class": st.CLASS_NAMES[c], "harmonic_energy_share": share,
                       "fundamental_prominence": prom, "mean_abs_pixel_diff": float(d[sel].mean())})
    write_csv(out / "per_class_signal.csv", f5)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True, layout="constrained")
    for c, ax in enumerate(axes.flat):
        for s in st.MAIN_SCANS:
            ax.semilogy(freq[1:], cls_psd[s, c][1:], color=st.SCAN_COLORS[s], lw=1,
                        label=st.SCAN_LABELS[s])
        ax.set_title(st.CLASS_NAMES[c], fontsize=10)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("Frequency (cycles / token)")
    fig.supylabel("Mean PSD (log)")
    fig.suptitle("F5a. Mean spectrum per class", fontsize=13)
    st.save(fig, fig_dir / "F5a_per_class_psd.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, key, title, fmt in (
            (axes[0], "harmonic_energy_share", "Energy at period-14 harmonics (%)", "{:.0f}"),
            (axes[1], "mean_abs_pixel_diff", "Mean |pixel change| (lower = smoother)", "{:.3f}")):
        mat = np.array([[next(r[key] for r in f5 if r["scan"] == s and r["class"] == n)
                         for s in st.MAIN_SCANS] for n in st.CLASS_NAMES])
        shown = mat * 100 if key == "harmonic_energy_share" else mat
        im = ax.imshow(shown, cmap="viridis", aspect="auto")
        for i in range(10):
            for j in range(len(st.MAIN_SCANS)):
                ax.text(j, i, fmt.format(shown[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if shown[i, j] < shown.mean() else "black")
        ax.set_xticks(range(len(st.MAIN_SCANS)), [st.SCAN_LABELS[s] for s in st.MAIN_SCANS],
                      rotation=20)
        ax.set_yticks(range(10), st.CLASS_NAMES)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("F5b. Do scan-induced signal properties differ by class?", fontsize=13)
    st.save(fig, fig_dir / "F5b_per_class_heatmaps.png")

    # ---------- F6: Spiral vs Snake ----------
    f6_scans = ("raster", "snake", "spiral", "spiral_inward")
    thirds = np.array_split(np.arange(N), 3)
    f6 = []
    for s in f6_scans:
        d = np.abs(np.diff(seqs[s], axis=1))
        for name, idx in zip(("early", "middle", "late"), thirds):
            tr = idx[idx < N - 1]
            row = {"scan": s, "segment": name, "tokens": f"{idx[0]}-{idx[-1]}",
                   "mean_intensity": float(seqs[s][:, idx].mean()),
                   "mean_abs_pixel_diff": float(d[:, tr].mean()),
                   "mean_centre_distance": float(np.linalg.norm(xy[s][idx] - CENTRE, axis=1).mean())}
            for c in range(10):
                row[f"intensity_{st.CLASS_NAMES[c]}"] = float(seqs[s][labels == c][:, idx].mean())
            f6.append(row)
    write_csv(out / "visit_order_segments.csv", f6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    for s in f6_scans:
        kw = dict(color=st.SCAN_COLORS[s], lw=1.3, label=st.SCAN_LABELS[s])
        axes[0].plot(np.linalg.norm(xy[s] - CENTRE, axis=1), **kw)
        axes[1].plot(seqs[s].mean(0), **kw)
    for ax in axes[:2]:
        for b in (thirds[1][0], thirds[2][0]):
            ax.axvline(b, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("Visit order (token index)")
    axes[0].set(ylabel="Distance to image centre (pixels)", title="When is the centre / edge visited?")
    axes[1].set(ylabel="Mean pixel intensity (60,000 images)", title="When is the garment visited?")
    x = np.arange(3)
    for k, s in enumerate(f6_scans):
        vals = [r["mean_intensity"] for r in f6 if r["scan"] == s]
        axes[2].bar(x + (k - 1.5) * 0.2, vals, 0.2, color=st.SCAN_COLORS[s],
                    label=st.SCAN_LABELS[s])
    axes[2].set_xticks(x, ["early 1/3", "middle 1/3", "late 1/3"])
    axes[2].set(ylabel="Mean intensity in segment", title="How much information arrives in each segment")
    handles, names = axes[2].get_legend_handles_labels()
    fig.legend(handles, [n + (" (overlaps Raster in left panel)" if n == "Snake" else "") for n in names],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.suptitle("F6. Same spatial continuity (Snake = Spiral), different order of information",
                 fontsize=13)
    st.save(fig, fig_dir / "F6_visit_order.png")

    # ---------- checks and summary ----------
    for s in scans:
        metrics[s]["mean_abs_pixel_diff"] = float(np.abs(np.diff(seqs[s], axis=1)).mean())
    p_out, p_in = spectra["spiral", "boxcar"], spectra["spiral_inward", "boxcar"]
    checks["inward_vs_outward"] = {
        "mean_abs_pixel_diff": [metrics["spiral"]["mean_abs_pixel_diff"],
                                metrics["spiral_inward"]["mean_abs_pixel_diff"]],
        "max_relative_psd_difference": float(np.max(np.abs(p_out - p_in) / p_out.max())),
        "mean_coordinate_distance": [metrics["spiral"]["mean_coordinate_distance"],
                                     metrics["spiral_inward"]["mean_coordinate_distance"]]}
    checks["handoff_reference"] = {
        s: {"mean_coordinate_distance": round(metrics[s]["mean_coordinate_distance"], 3),
            "mean_abs_pixel_diff": round(metrics[s]["mean_abs_pixel_diff"], 4)} for s in scans}
    checks["window4"] = {s: {"nonadjacent_moves": metrics[s]["nonadjacent_moves"],
                             "jump_window_fraction": round(metrics[s]["jump_window_fraction_w4"], 3)}
                         for s in scans}
    (out / "scan_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent / "phase2_t5"
    p.add_argument("--data", type=Path, default=root / "data/FashionMNIST/raw")
    p.add_argument("--output", type=Path, default=root / "results/train-local")
    p.add_argument("--figures", type=Path, default=root / "figures")
    main(p.parse_args())
