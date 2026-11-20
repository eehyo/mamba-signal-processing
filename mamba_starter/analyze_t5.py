"""Analyze scan geometry and Fashion-MNIST pixel sequences."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

from t5_preprocess import downsample, read_images
from t5_scan import DEFAULT_SEED, SCANS, SIZE, apply_scan, scan_indices


def summary(values):
    return {
        "mean": float(values.mean(dtype=np.float64)),
        "median": float(np.median(values)),
        "std": float(values.std(dtype=np.float64)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def psd(values):
    return periodogram(values, fs=1, window="boxcar", detrend="constant", scaling="density")


def analyze(args):
    images = downsample(read_images(args.data, args.split))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    fig_path, path_axes = plt.subplots(1, len(SCANS), figsize=(16, 3.5), layout="constrained")
    fig_dist, dist_axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    fig_psd, psd_axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    fig_repeat, repeat_ax = plt.subplots(figsize=(9, 4), layout="constrained")
    stats_rows, spectra, geometry, peak_rows, repeats = [], [], [], [], []
    orders = {}

    for scan, ax in zip(SCANS, path_axes):
        indices = scan_indices(scan, direction=args.direction, seed=args.seed)
        orders[scan] = indices
        row, col = np.divmod(indices, SIZE)
        name = f"{scan} ({args.direction})" if scan == "axis_aligned" else scan
        ax.plot(col, row, color="0.65", lw=0.6)
        dots = ax.scatter(col, row, c=np.arange(indices.size), cmap="viridis", s=10)
        ax.set(title=name, aspect="equal", xlabel="Column", ylabel="Row")
        ax.invert_yaxis()

        steps = np.diff(np.column_stack((row, col)), axis=0)
        distances = np.linalg.norm(steps, axis=1)
        sequences = apply_scan(images, indices)
        differences = np.abs(np.diff(sequences, axis=-1))
        for values, metric, axis, bins in (
            (distances, "coordinate_distance", dist_axes[0], np.arange(0, 19.6, 0.5)),
            (differences, "absolute_pixel_difference", dist_axes[1], np.linspace(0, 1, 51)),
        ):
            stats_rows.append({"scan": name, "metric": metric, "count": values.size,
                               **summary(values)})
            counts, edges = np.histogram(values, bins=bins)
            axis.stairs(counts / values.size, edges, label=name)

        frequency, power = psd(sequences)
        mean_power = power.mean(axis=0, dtype=np.float64)
        spectra.extend({"scan": name, "cycles_per_index": f, "mean_psd": p}
                       for f, p in zip(frequency, mean_power))
        psd_axes[0].plot(frequency[1:], mean_power[1:], label=name)
        for rank, ix in enumerate(np.argsort(-mean_power[1:], kind="stable")[:5] + 1, 1):
            peak_rows.append({"scan": name, "rank": rank, "cycles_per_index": frequency[ix],
                              "period_in_indices": 1 / frequency[ix], "mean_psd": mean_power[ix]})
            psd_axes[0].scatter(frequency[ix], mean_power[ix], color=psd_axes[0].lines[-1].get_color(),
                                s=12)

        coord_frequency, step_power = psd(steps.T)
        geometry.extend({"scan": name, "cycles_per_index": f, "row_step_psd": r, "col_step_psd": c}
                        for f, r, c in zip(coord_frequency, *step_power))
        psd_axes[1].plot(coord_frequency[1:], step_power[:, 1:].sum(axis=0), label=name)

        lags = np.arange(1, len(steps))
        agreement = [np.all(steps[:-lag] == steps[lag:], axis=1).mean() for lag in lags]
        repeats.extend({"scan": name, "lag": lag, "matching_step_fraction": fraction,
                        "pair_count": len(steps) - lag}
                       for lag, fraction in zip(lags, agreement))
        repeat_ax.plot(lags, agreement, label=name)

    fig_path.colorbar(dots, ax=list(path_axes), label="Visit index")
    for axis, label in zip(dist_axes, ("Euclidean distance (pixels)", "Absolute intensity difference")):
        axis.set(xlabel=label, ylabel="Fraction per bin")
        axis.legend()
    for axis, title in zip(psd_axes, ("Mean pixel sequence PSD", "Row + column step PSD")):
        axis.set(title=title, xlabel="Cycles per scan index", ylabel="Power spectral density")
        axis.legend()
    repeat_ax.set(xlabel="Lag (scan indices)", ylabel="Fraction of matching displacement vectors")
    repeat_ax.legend()
    for figure, filename in ((fig_path, "paths"), (fig_dist, "discontinuity"),
                             (fig_psd, "psd"), (fig_repeat, "path_periodicity")):
        figure.savefig(output / f"{filename}.png", dpi=160)
        plt.close(figure)

    for filename, rows in (("discontinuity", stats_rows), ("pixel_psd", spectra),
                           ("path_step_psd", geometry), ("top_pixel_frequencies", peak_rows),
                           ("path_periodicity", repeats)):
        write_csv(output / f"{filename}.csv", rows)
    np.savez(output / "scan_indices.npz", **orders)
    metadata = {"split": args.split, "image_count": len(images), "shape": [SIZE, SIZE],
                "axis_direction": args.direction, "random_seed": args.seed, "block_size": [2, 2],
                "spiral_start_row_col": [6, 6], "numpy": np.__version__}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/FashionMNIST/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--direction", choices=("raster", "column"), default="raster")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    analyze(parser.parse_args())
