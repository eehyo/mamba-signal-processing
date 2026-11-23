"""Shared figure style for Phase 2 T5 presentation figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Condition key -> display label. Keys match train.py --scan names.
SCAN_LABELS = {
    "raster": "Raster",
    "snake": "Snake",
    "random": "Random",
    "blockwise": "Blockwise",
    "spiral": "Spiral (out)",
    "spiral_inward": "Spiral (in)",
}
MAIN_SCANS = ("raster", "snake", "random", "blockwise", "spiral")
ALL_SCANS = MAIN_SCANS + ("spiral_inward",)
# t5_scan.scan_indices condition name for each key.
SCAN_CONDITION = {"raster": "axis_aligned", "snake": "snake", "random": "random",
                  "blockwise": "blockwise", "spiral": "spiral", "spiral_inward": "spiral_inward"}

# Colour-blind safe (Okabe-Ito); inward spiral shares the spiral hue family.
SCAN_COLORS = {
    "raster": "#0072B2",
    "snake": "#009E73",
    "random": "#7F7F7F",
    "blockwise": "#E69F00",
    "spiral": "#CC79A7",
    "spiral_inward": "#8E3B74",
}
MODEL_COLORS = {"mamba": "#D55E00", "cnn": "#56B4E9"}
MODEL_LABELS = {"mamba": "Mamba", "cnn": "1D CNN"}

CLASS_NAMES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


def setup():
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
