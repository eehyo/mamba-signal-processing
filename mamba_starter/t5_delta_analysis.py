"""F16-F17: what the trained selective SSM does with Δ, and how it differs from an LTI SSM.

F16a  meaning and limits of Δ: distribution, memory length implied by Δ·A (not Δ alone),
      dependence on token content
F16b  link to the research question: where Δ rises on the image, along the scan, and right
      after a spatial jump (background tokens only, to separate jumps from content)
F17   convolution view: an LTI SSM applies one fixed kernel at every position; the selective
      SSM's effective kernel changes token by token

Uses recorded Δ, A, B, C from MambaBlock.set_record on validation images (no test data).
Kernels are for the SSM sublayer only (input u after conv+SiLU), like MambaBlock.lti_kernel.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import t5_plot_style as st
from data import Standardizer, image_sequences, load_fashion
from evaluate import load_checkpoint
from t5_scan import SIZE, scan_indices

PREFERENCE = ("raster", "snake", "spiral", "blockwise", "random", "spiral_inward")
LN2 = np.log(2)


def find_checkpoints(roots, seed, allow_partial):
    found = {}
    for root in roots:
        for path in sorted(Path(root).glob(f"**/seed_{seed}/best.pt")):
            if not allow_partial and not (path.parent / "validation.json").exists():
                continue
            cfg = json.loads((path.parent.parent / "config.json").read_text())
            if cfg.get("dataset") != "fashion" or cfg["model"] != "mamba" or cfg.get("d_conv", 4) != 4:
                continue
            found.setdefault(cfg["scan"], path)
    return found


@torch.no_grad()
def record(path, images, labels, n, batch=250):
    model, ckpt = load_checkpoint(path, "cpu")
    cfg = ckpt["config"]
    idx = np.asarray(ckpt["split"]["val_indices"])[:n]
    raw = image_sequences(images[idx], cfg).astype(np.float32)          # [n,196,1] in [0,1]
    x = Standardizer.from_state(ckpt["normalizer"]).transform(raw).astype(np.float32)
    model.set_record(True)
    rec = [{"delta": [], "B": [], "C": []} for _ in model.layers]
    for i in range(0, n, batch):
        model(torch.from_numpy(x[i:i + batch]))
        for r, layer in zip(rec, model.records):
            for k in r:
                r[k].append(layer[k].numpy())
    out = []
    for r, layer in zip(rec, model.layers):
        out.append({k: np.concatenate(v) for k, v in r.items()}
                   | {"A": layer.mixer.A.detach().numpy(), "D": layer.mixer.D.detach().numpy()})
    order = scan_indices("axis_aligned" if cfg["scan"] == "raster" else cfg["scan"])
    return {"layers": out, "raw": raw[..., 0], "labels": labels[idx], "order": order,
            "scan": cfg["scan"], "path": str(path), "epoch": ckpt["best_validation"]["epoch"],
            "val_f1": ckpt["best_validation"]["macro_f1"]}


def rel_log_delta(delta):
    """log Δ relative to each channel's median, averaged over channels: [n,196]."""
    ld = np.log(delta)
    return (ld - np.median(ld.reshape(-1, ld.shape[-1]), axis=0)).mean(-1)


# ------------------------------------------------------------------ F16a
def f16a(r, fig_dir, rows):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
    for li, L in enumerate(r["layers"]):
        d = L["delta"].ravel()
        axes[0].hist(np.log10(d), bins=80, density=True, alpha=0.55, label=f"layer {li + 1}",
                     color=["#D55E00", "#0072B2"][li])
        rows.append({"scan": r["scan"], "layer": li + 1, "delta_median": float(np.median(d)),
                     "delta_p05": float(np.quantile(d, 0.05)), "delta_p95": float(np.quantile(d, 0.95))})
    axes[0].set(xlabel="log10 Δ (all tokens × 64 channels)", ylabel="density",
                title="Learned Δ spans orders of magnitude")
    axes[0].legend()

    L = r["layers"][0]
    absA = -L["A"]                                     # [h,n] > 0
    hl = {"slowest state (min |A|)": LN2 / (L["delta"] * absA.min(1)),
          "median state": LN2 / (L["delta"] * np.median(absA, 1)),
          "fastest state (max |A|)": LN2 / (L["delta"] * absA.max(1))}
    bins = np.logspace(-2, 4, 90)
    for (name, v), c in zip(hl.items(), ("#009E73", "#999999", "#CC79A7")):
        axes[1].hist(np.clip(v.ravel(), 1e-2, 1e4), bins=bins, density=True, histtype="step", lw=1.8,
                     color=c, label=name)
        rows.append({"scan": r["scan"], "layer": 1, "state": name,
                     "half_life_median_tokens": float(np.median(v))})
    for x, lab in ((4, "conv window (4)"), (196, "sequence (196)")):
        axes[1].axvline(x, color="k", ls=":", lw=1)
        axes[1].text(x, axes[1].get_ylim()[1] * 0.92, lab, rotation=90, ha="right", va="top", fontsize=8)
    axes[1].set(xscale="log", xlabel="memory half-life ln2 / (Δ·|A_n|)  [tokens]", ylabel="density",
                title="Same Δ, different memory: Δ·A decides (layer 1)")
    axes[1].legend(fontsize=8, loc="upper left")

    raw = r["raw"]
    edges = np.linspace(0, 1, 11)
    centre = (edges[:-1] + edges[1:]) / 2
    for li, Lx in enumerate(r["layers"]):
        rel = rel_log_delta(Lx["delta"])
        b = np.clip(np.digitize(raw, edges) - 1, 0, 9)
        m = np.array([rel[b == k].mean() if (b == k).any() else np.nan for k in range(10)])
        axes[2].plot(centre, m, "-o", color=["#D55E00", "#0072B2"][li], label=f"layer {li + 1}")
    axes[2].axhline(0, color="k", lw=0.7)
    axes[2].set(xlabel="pixel intensity of the current token (0 = background)",
                ylabel="mean log Δ − channel median",
                title="Δ depends on content (selectivity)")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"F16a. What the learned step size Δ means — and its limits   "
                 f"[{st.SCAN_LABELS.get(r['scan'], r['scan'])} model, {len(raw):,} validation images]",
                 fontsize=12.5)
    fig.text(0.5, -0.04, "Limits: Δ is computed from the conv-mixed input (current + 3 previous tokens), "
             "not from one pixel; layer 2 sees mixed features; memory length needs Δ·A, and B, C, the gate and "
             "the residual path also shape the output.", ha="center", fontsize=8.5, color="0.35")
    st.save(fig, fig_dir / "F16a_delta_meaning.png")


# ------------------------------------------------------------------ F16b
def f16b(recs, fig_dir, rows):
    scans = [s for s in st.ALL_SCANS if s in recs]
    fig = plt.figure(figsize=(16, 8.2), layout="constrained")
    gs = fig.add_gridspec(2, max(len(scans), 3))
    maps, lim = {}, 0
    for s in scans:
        r = recs[s]
        rel = rel_log_delta(r["layers"][0]["delta"]).mean(0)          # per sequence position
        grid = np.zeros(SIZE * SIZE)
        grid[r["order"]] = rel
        maps[s] = grid.reshape(SIZE, SIZE)
        lim = max(lim, np.abs(maps[s]).max())
    for j, s in enumerate(scans):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(maps[s], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set(title=st.SCAN_LABELS[s], xticks=[], yticks=[])
    fig.colorbar(im, ax=fig.axes[:len(scans)], shrink=0.8, label="layer-1 log Δ − channel median\n(red = forgets faster)")

    ax = fig.add_subplot(gs[1, :2])
    for s in scans:
        rel = rel_log_delta(recs[s]["layers"][0]["delta"]).mean(0)
        ax.plot(rel, color=st.SCAN_COLORS[s], lw=1.2, label=st.SCAN_LABELS[s],
                ls="--" if s == "spiral_inward" else "-")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="position in the scan (token index)", ylabel="mean layer-1 log Δ − median",
           title="Δ along the sequence")
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[1, 2:])
    labels, means, errs, colors = [], [], [], []
    for s in scans:
        r = recs[s]
        rc, cc = np.divmod(r["order"], SIZE)
        dist = np.r_[0, np.hypot(np.diff(rc), np.diff(cc))]          # move into each token
        jump = dist > 1 + 1e-9
        if jump.sum() < 5 or (~jump[1:]).sum() < 5:
            continue
        rel = rel_log_delta(r["layers"][0]["delta"])                 # [n,196]
        raw = r["raw"]
        bg = raw < 0.02
        prev_bg = np.c_[np.zeros((len(raw), 1), bool), raw[:, :-1] < 0.02]   # excludes token 0
        ok = bg & prev_bg                                              # background → background moves
        for flag, name in ((False, "adjacent move"), (True, "after a spatial jump")):
            sel = ok & (jump[None, :] == flag)
            v = rel[sel]
            labels.append(f"{st.SCAN_LABELS[s]}\n{name}")
            means.append(v.mean())
            errs.append(1.96 * v.std() / np.sqrt(len(v)))
            colors.append(st.SCAN_COLORS[s])
            rows.append({"scan": s, "move": name, "tokens": int(sel.sum()), "mean_rel_log_delta": float(v.mean())})
    if labels:
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=errs, color=colors, alpha=0.8, capsize=3,
               hatch=["", "//"] * (len(labels) // 2), edgecolor="white")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set(ylabel="layer-1 log Δ − median (background tokens only)",
               title="Does the model 'reset' after a scan jump? (content held fixed)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Needs a scan with spatial jumps (Raster or Blockwise checkpoint)",
                ha="center", va="center", fontsize=10, color="0.4")
    fig.suptitle("F16b. Where the trained model raises Δ (forgets faster): on the image, along the scan, "
                 "after jumps   [seed 0 of each scan, layer 1]", fontsize=12.5)
    st.save(fig, fig_dir / "F16b_delta_vs_scan.png")


# ------------------------------------------------------------------ F17
def effective_kernels(delta, B, C, A, S):
    """Selective SSM effective kernel K[t, s] (lag s) for one image and channel; NaN for s > t."""
    T = len(delta)
    P = np.cumsum(delta)
    K = np.full((T, S), np.nan)
    for t in range(T):
        for s in range(min(S, t + 1)):
            k = t - s
            decay = np.exp(A * (P[t] - P[k]))                         # [n]
            K[t, s] = (C[t] * decay * delta[k] * B[k]).sum()
    return K


def f17(r, fig_dir, rows, S=40):
    L = r["layers"][0]
    labels = r["labels"]
    i = int(np.flatnonzero(labels == 4)[0]) if (labels == 4).any() else 0   # a Coat
    d_img = L["delta"][i]                                              # [196,64]
    c = int(np.argmax(np.log(d_img).std(0)))                            # most selective channel here
    A = L["A"][c]
    Ksel = effective_kernels(d_img[:, c], L["B"][i], L["C"][i], A, S)
    dbar = L["delta"][:, :, c].mean()
    Bbar, Cbar = L["B"].mean((0, 1)), L["C"].mean((0, 1))
    klti = np.array([(Cbar * np.exp(s * dbar * A) * dbar * Bbar).sum() for s in range(S)])
    Klti = np.where(np.arange(S)[None, :] <= np.arange(len(d_img))[:, None], klti[None, :], np.nan)

    fig = plt.figure(figsize=(16, 8.4))          # manual layout: keeps the heatmaps wide
    a0 = fig.add_axes([0.05, 0.50, 0.022, 0.40])
    T = len(d_img)
    a0.pcolormesh([0, 1], np.arange(T + 1) - 0.5, r["raw"][i][:, None], cmap="gray_r", vmin=0, vmax=1)
    a0.invert_yaxis()
    a0.set(xticks=[], ylabel="output position t (scan order)", title="pixel")
    lim = np.nanquantile(np.abs(np.r_[Ksel[~np.isnan(Ksel)], klti]), 0.99)
    for col, (K, title) in enumerate(((Klti, "LTI: one fixed kernel\n(every row identical = convolution)"),
                                      (Ksel, "Selective: kernel changes per token\n(rows differ = not a convolution)"))):
        ax = fig.add_axes([0.105 + col * 0.25, 0.50, 0.21, 0.40])
        im = ax.pcolormesh(np.arange(S + 1) - 0.5, np.arange(T + 1) - 0.5, np.ma.masked_invalid(K),
                           cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.invert_yaxis()
        ax.set(xlabel="lag s (how many tokens back)", title=title)
        ax.tick_params(labelleft=(col == 0))
    fig.colorbar(im, cax=fig.add_axes([0.575, 0.52, 0.008, 0.36]),
                 label="weight of input u_(t−s) in output y_t")

    ax = fig.add_axes([0.68, 0.50, 0.30, 0.40])
    valid = np.arange(S, len(d_img))
    raw_i = r["raw"][i]
    def pick(cond, name):
        cand = valid[cond(raw_i[valid])]
        return (name, int(cand[len(cand) // 2])) if len(cand) else None
    picks = dict(x for x in (pick(lambda v: v < 0.02, "reading background"),
                             pick(lambda v: (v > 0.15) & (v < 0.45), "garment edge"),
                             pick(lambda v: v > 0.6, "inside the garment")) if x)
    for (name, t), colr in zip(picks.items(), ("#009E73", "#999999", "#D55E00")):
        ax.plot(Ksel[t], color=colr, lw=1.6, label=f"selective, t={t}: {name} (Δ_t={d_img[t, c]:.1e})")
        rows.append({"scan": r["scan"], "channel": c, "t": int(t), "case": name,
                     "delta": float(d_img[t, c]), "kernel_l1": float(np.nansum(np.abs(Ksel[t])))})
    ax.plot(klti, "k--", lw=1.6, label="LTI counterpart (same at every t)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="lag s", ylabel="kernel weight", title="Kernels at three positions")
    ax.legend(fontsize=7.5)

    ax = fig.add_axes([0.05, 0.07, 0.88, 0.31])
    ax.fill_between(np.arange(len(d_img)), 0, r["raw"][i], color="0.85", step="mid", label="pixel intensity")
    ax.set(xlabel="position in the scan (token index)", ylabel="pixel intensity", xlim=(0, len(d_img) - 1))
    ax2 = ax.twinx()
    ax2.semilogy(d_img[:, c], color="#D55E00", lw=1.4, label=f"Δ_t of channel {c}")
    ax2.axhline(dbar, color="k", ls="--", lw=1, label="LTI: one constant Δ")
    ax2.set_ylabel("Δ_t (log)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.set_ylim(0, 1.3)  # headroom so the legend sits above the data
    lo, hi = ax2.get_ylim()
    ax2.set_ylim(lo, hi * 10)
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", ncol=3, framealpha=1)
    ax.set_title("Why selectivity matters: Δ (forget rate) reacts to what is being read, so the model can "
                 "keep or drop context depending on content — an LTI filter cannot")
    fig.suptitle(f"F17. LTI vs selective SSM in the convolution view   "
                 f"[{st.SCAN_LABELS.get(r['scan'], r['scan'])} model, layer 1, channel {c}, one Coat image]",
                 fontsize=12.5)
    fig.text(0.5, 0.0, "SSM sublayer only (input u = conv+SiLU output; D skip omitted). LTI counterpart = "
             "same A with Δ, B, C fixed at their averages over validation tokens, for illustration.",
             ha="center", fontsize=8.5, color="0.35")
    st.save(fig, fig_dir / "F17_lti_vs_selective.png")


def write(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


def main():
    root = Path(__file__).resolve().parent.parent / "phase2_t5"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", type=Path, nargs="+", default=[
        root / "outputs/t5-mamba-validation", root / "outputs/t5-mamba-validation-extra"])
    p.add_argument("--data_root", type=Path, default=root / "data/FashionMNIST/raw")
    p.add_argument("--figures", type=Path, default=root / "figures")
    p.add_argument("--tables", type=Path, default=root / "results/delta-analysis")
    p.add_argument("--images", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_partial", action="store_true",
                   help="also use best.pt of runs still training (dry runs only)")
    args = p.parse_args()
    torch.set_num_threads(4)
    st.setup()
    args.tables.mkdir(parents=True, exist_ok=True)
    ckpts = find_checkpoints(args.roots, args.seed, args.allow_partial)
    if not ckpts:
        raise SystemExit("no Mamba checkpoints found")
    images, labels = load_fashion(args.data_root, "train")
    recs = {}
    for scan, path in ckpts.items():
        recs[scan] = record(path, images, labels, args.images)
        print(f"{scan}: {path} (best epoch {recs[scan]['epoch']}, val F1 {recs[scan]['val_f1']:.4f})", flush=True)
    main_scan = next(s for s in PREFERENCE if s in recs)
    a, b, c = [], [], []
    f16a(recs[main_scan], args.figures, a)
    f16b(recs, args.figures, b)
    f17(recs[main_scan], args.figures, c)
    write(args.tables / "delta_summary.csv", a)
    write(args.tables / "delta_by_scan.csv", b)
    write(args.tables / "kernels.csv", c)
    (args.tables / "sources.json").write_text(json.dumps(
        {s: {"checkpoint": os.path.relpath(r["path"], args.tables), "best_epoch": r["epoch"], "val_macro_f1": r["val_f1"]}
         for s, r in recs.items()}, indent=2) + "\n")
    print("main model for F16a/F17:", main_scan)


if __name__ == "__main__":
    main()
