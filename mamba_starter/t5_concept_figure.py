"""F0: how the T5 Mamba classifier reads a scanned image (schematic, no data)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import t5_plot_style as st


def box(ax, x, y, w, h, text, fc, fs=9, ec="0.3"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                 color=kw.pop("color", "0.25"), **kw))


def main(out):
    st.setup()
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set(xlim=(0, 14), ylim=(0, 6.4))
    ax.axis("off")

    # 1. image -> scan -> tokens
    for r in range(4):
        for c in range(4):
            ax.add_patch(Rectangle((0.3 + c * 0.3, 4.3 + r * 0.3), 0.3, 0.3,
                                   fc=str(0.35 + 0.15 * ((r + c) % 3)), ec="white"))
    ax.text(0.9, 5.65, "14×14 garment image", ha="center", fontsize=9)
    arrow(ax, (1.6, 4.9), (2.4, 4.9))
    ax.text(2.0, 5.15, "scan order", ha="center", fontsize=8, color="#CC79A7", weight="bold")
    n, x0, w = 12, 2.6, 0.62
    for i in range(n):
        fc = "#FFE0B2" if 5 <= i <= 8 else "#E3F2FD"
        ax.add_patch(Rectangle((x0 + i * w, 4.6), w * 0.9, 0.6, fc=fc, ec="0.4"))
        ax.text(x0 + i * w + w * 0.45, 4.9, f"x{i+1}" if i < 10 else ("…" if i == 10 else "x196"),
                ha="center", va="center", fontsize=8)
    ax.text(x0 + 6 * w, 5.45, "196 tokens (one pixel = one token)", ha="center", fontsize=9)

    # 2. causal conv window
    ax.add_patch(Rectangle((x0 + 5 * w - 0.05, 4.5), 4 * w + 0.03, 0.8, fill=False, ec="#D55E00", lw=2))
    ax.text(x0 + 7 * w, 4.25, "causal conv (d_conv=4): mixes the current token + the previous 3 only\n"
            "→ matters whether sequence neighbours are also image neighbours (F2)",
            ha="center", va="top", fontsize=8.5, color="#D55E00")

    # 3. selective state update chain
    y = 2.1
    for i in range(6):
        cx = 3.0 + i * 1.45
        box(ax, cx, y, 1.0, 0.7, f"h{i+1}" if i < 5 else "h196", "#E8F5E9", fs=10)
        if i < 5:
            arrow(ax, (cx + 1.0, y + 0.35), (cx + 1.45, y + 0.35), color="#009E73")
        arrow(ax, (cx + 0.5, 3.0), (cx + 0.5, y + 0.72), color="0.5")
    ax.text(2.85, y + 0.35, "state\n(memory)", ha="right", va="center", fontsize=9, color="#009E73",
            weight="bold")
    ax.text(7.0, 1.55, "h_t = exp(Δ_t·A)·h_(t-1) + Δ_t·B_t·x_t      "
            "Δ_t, B_t, C_t are computed from the current input (selective)\n"
            "large Δ → forget the past quickly, weight the current input   /   small Δ → keep the past",
            ha="center", va="top", fontsize=9, color="#1B5E20")
    ax.text(7.0, 0.75, "Earlier tokens shape how later tokens are read → the same path read in reverse (Spiral out/in) can give different results (F15)",
            ha="center", fontsize=8.5, color="#8E3B74")

    # 4. mean pooling + head
    box(ax, 11.9, 2.0, 1.8, 0.9, "average of all\noutputs (mean pool)", "#FFF3E0")
    arrow(ax, (11.3, 2.45), (11.9, 2.45))
    ax.text(11.6, 2.75, "all\noutputs", ha="center", va="bottom", fontsize=7)
    box(ax, 11.9, 3.4, 1.8, 0.8, "classify into\n1 of 10 classes", "#FCE4EC")
    arrow(ax, (12.8, 2.9), (12.8, 3.4))
    ax.text(12.8, 1.75, "averages every token,\nnot just the last one", ha="center", va="top", fontsize=8)

    ax.text(0.2, 6.25, "F0. How the T5 Mamba classifier reads an image (2 layers, d_model=32, d_state=16)",
            fontsize=13, weight="bold")
    ax.text(0.2, 3.3, "2 layers:\nconv reach\nstacks up to\n7 tokens", fontsize=8, color="0.35")
    st.save(fig, out)


def block_diagram(out):
    """F0b: every component of the classifier and one Mamba block, with T5 shapes."""
    st.setup()
    fig, ax = plt.subplots(figsize=(16, 9.6))
    ax.set(xlim=(0, 16), ylim=(0, 9.6))
    ax.axis("off")
    role = dict(ha="center", va="top", fontsize=7.6, color="0.35")
    C_IO, C_NORM, C_LIN, C_CONV, C_SSM, C_GATE = ("#ECEFF1", "#E1F5FE", "#FFF8E1", "#FFE0B2",
                                                  "#E8F5E9", "#F3E5F5")

    # ---- classifier stack (left): role text inside each box so arrows never cross it
    ax.text(1.55, 8.95, "Classifier", ha="center", fontsize=11, weight="bold")
    stack = [("pixel tokens [196, 1]", C_IO, "scanned image"),
             ("input_proj 1 → 32", C_LIN, "lift each pixel to 32 features"),
             ("Mamba block 1", C_SSM, "local mixing + memory"),
             ("Mamba block 2", C_SSM, "mixes block-1 features"),
             ("RMSNorm", C_NORM, "rescale features"),
             ("mean pool (196 tokens)", C_GATE, "every token counts"),
             ("head 32 → 10", C_LIN, "class scores")]
    ys = [7.95, 6.75, 5.55, 4.35, 3.15, 1.95, 0.75]
    for (t, c, r), y in zip(stack, ys):
        ax.add_patch(FancyBboxPatch((0.4, y), 2.3, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=c, ec="#2E7D32" if "block" in t else "0.3", lw=1))
        ax.text(1.55, y + 0.52, t, ha="center", va="center", fontsize=8.8)
        ax.text(1.55, y + 0.22, r, ha="center", va="center", fontsize=7.4, color="0.35")
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (1.55, a), (1.55, b + 0.84))
    for y_from, y_to in ((6.35, 7.9), (5.55, 0.45)):
        ax.annotate("", xy=(3.55, y_to), xytext=(2.72, y_from),
                    arrowprops=dict(arrowstyle="-", ls="--", color="#2E7D32"))

    # ---- one block (right)
    ax.add_patch(FancyBboxPatch((3.55, 0.45), 12.2, 7.45, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="none", ec="#2E7D32", lw=1.5, ls="--"))
    ax.text(3.75, 7.6, "Inside one Mamba block:  x + Mixer(RMSNorm(x))      d=32, h=64, n=16",
            fontsize=11, weight="bold", color="#2E7D32")

    def node(x, y, w, t, c, r=None, h=0.62, fs=8.5, above=False):
        box(ax, x, y, w, h, t, c, fs=fs)
        if r and above:
            ax.text(x + w / 2, y + h + 0.05, r, **{**role, "va": "bottom"})
        elif r:
            ax.text(x + w / 2, y - 0.05, r, **role)

    y0 = 6.3
    node(3.85, y0, 1.3, "x\n[196, 32]", C_IO)
    node(5.55, y0, 1.3, "RMSNorm", C_NORM, "normalise\neach token")
    node(7.25, y0, 1.6, "in_proj\n32 → 2×64", C_LIN, "widen, split into\nu (main) and z (gate)",
         above=True)
    arrow(ax, (5.15, y0 + 0.31), (5.55, y0 + 0.31))
    arrow(ax, (6.85, y0 + 0.31), (7.25, y0 + 0.31))
    node(9.3, y0, 1.0, "u [196, 64]", C_IO, fs=8)
    arrow(ax, (8.85, y0 + 0.31), (9.3, y0 + 0.31))
    node(10.7, y0, 1.9, "causal conv1d\nk=4, per channel", C_CONV,
         "current + previous 3 tokens\n(scan-neighbour window, F2)")
    arrow(ax, (10.3, y0 + 0.31), (10.7, y0 + 0.31))
    node(13.0, y0, 0.9, "SiLU", C_NORM)
    arrow(ax, (12.6, y0 + 0.31), (13.0, y0 + 0.31))
    arrow(ax, (13.45, y0), (13.45, 4.84))

    # selective SSM sub-box, kept right of the z line
    ax.add_patch(FancyBboxPatch((9.0, 2.2), 6.45, 3.25, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#F1F8E9", ec="#558B2F", lw=1.2))
    ax.text(9.15, 5.2, "Selective SSM: Δ, B, C computed from the input at every token",
            fontsize=9.3, weight="bold", color="#33691E")
    node(12.75, 4.2, 1.4, "x_proj\n64 → 2+16+16", C_LIN, fs=8)
    node(10.95, 4.2, 1.45, "dt_proj 2 → 64\n+ softplus", C_LIN, fs=8)
    arrow(ax, (12.75, 4.51), (12.4, 4.51))
    ax.text(12.57, 4.86, "Δ raw", fontsize=6.5, ha="center", va="bottom")
    node(9.2, 4.2, 1.55, "Δ_t [64] > 0\nstep / forget rate", C_GATE, fs=8)
    arrow(ax, (10.95, 4.51), (10.75, 4.51))
    ax.text(14.25, 4.12, "B_t, C_t [16]\nwrite / read", fontsize=7.5, ha="left", va="top",
            color="#33691E")
    box(ax, 9.2, 2.4, 6.05, 1.2,
        "h_t = exp(Δ_t·A) ⊙ h_(t−1) + Δ_t·B_t·u_t        y_t = C_t·h_t + D·u_t\n"
        "A = −exp(A_log) [64×16] is fixed (does not depend on the input)\n"
        "decay per state exp(Δ_t·A_n): 16 states = a bank of EMAs with different memory lengths",
        "#FFFFFF", fs=8, ec="#558B2F")
    arrow(ax, (9.97, 4.2), (9.97, 3.62), color="#558B2F")
    arrow(ax, (13.45, 4.2), (13.45, 3.62), color="#558B2F")

    # z path: down the left of the SSM box, then along the bottom row
    ax.plot([8.6, 8.6], [y0, 1.31], color="0.25", lw=1.2)
    arrow(ax, (8.6, 1.31), (9.0, 1.31))
    node(9.0, 1.0, 1.0, "z [196, 64]", C_IO, fs=8)
    node(10.4, 1.0, 1.0, "SiLU(z)", C_NORM, "gate")
    arrow(ax, (10.0, 1.31), (10.4, 1.31))
    node(11.85, 1.0, 0.75, "×", C_GATE, "y ⊙ gate", fs=12)
    arrow(ax, (11.4, 1.31), (11.85, 1.31))
    arrow(ax, (12.22, 2.4), (12.22, 1.62))
    node(13.0, 1.0, 1.25, "out_proj\n64 → 32", C_LIN, "back to d=32")
    arrow(ax, (12.6, 1.31), (13.0, 1.31))
    node(14.65, 1.0, 0.75, "+ x", C_IO, "residual", fs=10)
    arrow(ax, (14.25, 1.31), (14.65, 1.31))

    ax.text(3.85, 3.3, "Roles in one line\n"
            "• conv1d: short local filter over the scan\n"
            "• SSM: long memory, leaky per state\n"
            "• Δ_t: how much to forget / write now\n"
            "• B_t, C_t: what to write / read\n"
            "• z gate: per-channel output switch\n"
            "• residual + RMSNorm: stable stacking",
            fontsize=8.3, va="center", color="0.2",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FAFAFA", ec="0.8"))
    ax.text(0.3, 0.12, "Parameters: 20,330 in total (block 1: 9,920). Shapes are per image; the batch axis is "
            "omitted. Educational sequential scan (mamba_block.py), not the fused CUDA kernel.",
            fontsize=8, color="0.4")
    ax.text(0.3, 9.35, "F0b. Mamba block components and their roles (T5 configuration)",
            fontsize=13, weight="bold")
    st.save(fig, out)


if __name__ == "__main__":
    figs = Path(__file__).resolve().parent.parent / "phase2_t5/figures"
    main(figs / "F0_mamba_reading.png")
    block_diagram(figs / "F0b_mamba_block.png")
