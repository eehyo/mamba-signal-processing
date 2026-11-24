"""Summarise the official test evaluation (phase2_t5/outputs/t5-test, see FINAL_PROTOCOL.md).

Writes to phase2_t5/results/model-test/:
  performance.csv       mean ± sample std over seeds, test and validation side by side
  paired_vs_raster.csv  same-seed differences (scan − Raster), positive-seed counts
  spiral_direction.csv  same-seed Inward − Outward
  dconv_gap.csv         structured scan − Random gap at d_conv=4 vs d_conv=1
  per_class_f1.csv      per-class F1 from the confusion matrices, mean over seeds
  runs.csv              one row per checkpoint
Validation numbers come from each checkpoint's stored best_validation.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

T5 = Path(__file__).resolve().parent.parent / "phase2_t5"
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
SCANS = ["raster", "snake", "column", "random", "blockwise", "spiral", "spiral_inward"]
GROUPS = {"mamba": ("mamba", 4, "20 epochs"), "mamba_k1": ("mamba", 1, "20 epochs"),
          "cnn": ("cnn", None, "20 epochs"), "cnn60": ("cnn", None, "<=60 epochs, patience 5")}


def per_class_f1(cm):
    cm = np.asarray(cm, float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)


def mean_std(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=T5 / "outputs/t5-test")
    p.add_argument("--out", type=Path, default=T5 / "results/model-test")
    p.add_argument("--protocol", type=Path, default=T5 / "FINAL_PROTOCOL.md")
    args = p.parse_args()
    import hashlib
    sha = hashlib.sha256(args.protocol.read_bytes()).hexdigest()

    runs = defaultdict(dict)  # (group, scan) -> seed -> record
    rows = []
    for path in sorted(args.runs.glob("*/*/seed_*.json")):
        group, scan, seed = path.parent.parent.name, path.parent.name, int(path.stem.split("_")[1])
        r = json.loads(path.read_text())
        if r["split"] != "test" or r["protocol_sha256"] != sha or r["examples"] != 10000:
            raise SystemExit(f"{path}: not a test result under the current FINAL_PROTOCOL.md")
        ck = torch.load(r["checkpoint"], map_location="cpu", weights_only=False)
        if ck["config"]["scan"] != scan or ck["seed"] != seed:
            raise SystemExit(f"{path}: checkpoint is {ck['config']['scan']} seed {ck['seed']}")
        val = ck["best_validation"]
        rec = dict(acc=r["accuracy"], f1=r["macro_f1"], cm=r["confusion_matrix"],
                   val_acc=val["accuracy"], val_f1=val["macro_f1"], epoch=val["epoch"])
        runs[(group, scan)][seed] = rec
        rows.append(dict(group=group, model=GROUPS[group][0], d_conv=GROUPS[group][1] or "",
                         scan=scan, seed=seed, best_epoch=rec["epoch"],
                         test_accuracy=f"{rec['acc']:.4f}", test_macro_f1=f"{rec['f1']:.4f}",
                         val_accuracy=f"{rec['val_acc']:.4f}", val_macro_f1=f"{rec['val_f1']:.4f}"))
    write(args.out / "runs.csv", rows)

    def ordered(group):
        return [s for s in SCANS if (group, s) in runs]

    perf, paired, pcf = [], [], []
    for group in GROUPS:
        for scan in ordered(group):
            rs = runs[(group, scan)]
            seeds = sorted(rs)
            ta, sa = mean_std([rs[k]["acc"] for k in seeds])
            tf, sf = mean_std([rs[k]["f1"] for k in seeds])
            va, _ = mean_std([rs[k]["val_acc"] for k in seeds])
            vf, svf = mean_std([rs[k]["val_f1"] for k in seeds])
            model, k, budget = GROUPS[group]
            perf.append(dict(group=group, model=model, d_conv=k or "", training=budget, scan=scan,
                             seeds=len(seeds), test_accuracy_mean=f"{ta:.4f}", test_accuracy_std=f"{sa:.4f}",
                             test_macro_f1_mean=f"{tf:.4f}", test_macro_f1_std=f"{sf:.4f}",
                             val_accuracy_mean=f"{va:.4f}", val_macro_f1_mean=f"{vf:.4f}",
                             val_macro_f1_std=f"{svf:.4f}", test_minus_val_f1=f"{tf - vf:+.4f}"))
            f1c = np.mean([per_class_f1(rs[k]["cm"]) for k in seeds], axis=0)
            pcf.append(dict(group=group, scan=scan, **{c: f"{v:.4f}" for c, v in zip(CLASSES, f1c)}))
            if scan != "raster" and (group, "raster") in runs:
                base = runs[(group, "raster")]
                common = [k for k in seeds if k in base]
                for split, key in (("test", "f1"), ("validation", "val_f1")):
                    d = [rs[k][key] - base[k][key] for k in common]
                    paired.append(dict(group=group, scan=scan, split=split, mean_diff=f"{np.mean(d):+.4f}",
                                       per_seed=" ".join(f"{x:+.4f}" for x in d),
                                       positive=f"{sum(x > 0 for x in d)}/{len(d)}"))
    write(args.out / "performance.csv", perf)
    write(args.out / "paired_vs_raster.csv", paired)
    write(args.out / "per_class_f1.csv", pcf)

    direction = []
    for group in GROUPS:
        if (group, "spiral") in runs and (group, "spiral_inward") in runs:
            out, inw = runs[(group, "spiral")], runs[(group, "spiral_inward")]
            for split, key in (("test", "f1"), ("validation", "val_f1")):
                d = [inw[k][key] - out[k][key] for k in sorted(out) if k in inw]
                direction.append(dict(group=group, split=split, inward_minus_outward=f"{np.mean(d):+.4f}",
                                      per_seed=" ".join(f"{x:+.4f}" for x in d),
                                      inward_better=f"{sum(x > 0 for x in d)}/{len(d)}"))
    write(args.out / "spiral_direction.csv", direction)

    gap = []
    for scan in ["raster", "snake", "column", "blockwise", "spiral"]:
        row = dict(scan=scan)
        for group, k in (("mamba", 4), ("mamba_k1", 1)):
            if (group, scan) not in runs or (group, "random") not in runs:
                break
            for split, key in (("test", "f1"), ("validation", "val_f1")):
                d = [runs[(group, scan)][s][key] - runs[(group, "random")][s][key]
                     for s in sorted(runs[(group, scan)]) if s in runs[(group, "random")]]
                row[f"gap_k{k}_{split}"] = f"{np.mean(d):+.4f}"
                row[f"gap_k{k}_{split}_per_seed"] = " ".join(f"{x:+.4f}" for x in d)
        else:
            for split in ("test", "validation"):
                g4 = [float(x) for x in row[f"gap_k4_{split}_per_seed"].split()]
                g1 = [float(x) for x in row[f"gap_k1_{split}_per_seed"].split()]
                row[f"gap_growth_{split}"] = f"{np.mean(g1) - np.mean(g4):+.4f}"
                row[f"gap_grew_{split}"] = f"{sum(b > a for a, b in zip(g4, g1))}/{len(g4)}"
            gap.append(row)
    if gap:
        write(args.out / "dconv_gap.csv", gap)
    (args.out / "README.md").write_text(
        "# Official test results\n\n"
        f"Generated by `mamba_starter/t5_test_summary.py` from `outputs/t5-test` "
        f"(FINAL_PROTOCOL.md sha256 `{sha}`). Fashion-MNIST official test set, 10,000 images, "
        "evaluated once per checkpoint. Validation columns are each checkpoint's stored best validation.\n")
    for r in perf:
        print(f"{r['group']:9s} {r['scan']:14s} test F1 {r['test_macro_f1_mean']} ± {r['test_macro_f1_std']}"
              f"  (val {r['val_macro_f1_mean']}, {r['test_minus_val_f1']})")


if __name__ == "__main__":
    main()
