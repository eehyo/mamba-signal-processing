"""Dump per-image validation predictions for every T5 checkpoint under a run root.

For each <root>/**/seed_*/best.pt writes <out>/<model>_<scan>[_k<d_conv>]_seed<k>.npz with
y, pred, logits, val_index (indices into the official training file) and metadata,
then checks accuracy / macro-F1 against the stored seed_*/validation.json.
Validation only: the official test split is never read here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from data import SequenceDataset, collate_sequences
from evaluate import evaluation_data, load_checkpoint


def run_name(cfg, seed):
    name = f"{cfg['model']}_{cfg['scan']}"
    if cfg["model"] == "mamba" and cfg.get("d_conv", 4) != 4:
        name += f"_k{cfg['d_conv']}"
    return f"{name}_seed{seed}"


@torch.no_grad()
def predict(ckpt_path, data_root, device, batch_size):
    model, ckpt = load_checkpoint(ckpt_path, device)
    cfg = ckpt["config"]
    x, y = evaluation_data(ckpt, data_root, "validation")
    loader = DataLoader(SequenceDataset(x, y), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_sequences)
    logits = []
    for batch in loader:
        out = model(batch["x"].to(device), mask=batch["mask"].to(device))
        if not torch.isfinite(out).all():
            raise FloatingPointError(f"nonfinite logits in {ckpt_path}")
        logits.append(out.float().cpu())
    logits = torch.cat(logits).numpy()
    return cfg, ckpt, np.asarray(y), logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # validation.json is written only after a seed's last epoch: skip runs still training.
    ckpts = sorted(c for r in args.root for c in r.glob("**/seed_*/best.pt")
                   if (c.parent / "validation.json").exists())
    if not ckpts:
        raise SystemExit(f"no seed_*/best.pt under {args.root}")
    problems = []
    for path in ckpts:
        meta = torch.load(path, map_location="cpu", weights_only=True)
        name = run_name(meta["config"], meta["seed"])
        target = args.out / f"{name}.npz"
        if target.exists() and not args.overwrite:
            print(f"skip {name} (exists)")
            continue
        cfg, ckpt, y, logits = predict(path, args.data_root, args.device, args.batch_size)
        pred = logits.argmax(1)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)
        stored = ckpt["best_validation"]
        ok = abs(acc - stored["accuracy"]) < 1e-6 and abs(f1 - stored["macro_f1"]) < 1e-6
        np.savez_compressed(target, y=y, pred=pred, logits=logits,
                            val_index=np.asarray(ckpt["split"]["val_indices"]),
                            model=cfg["model"], scan=cfg["scan"], seed=ckpt["seed"],
                            d_conv=cfg.get("d_conv", 4), best_epoch=stored["epoch"],
                            checkpoint=str(path))
        print(f"{name}: n={len(y)} acc={acc:.4f} f1={f1:.4f} "
              f"(stored acc={stored['accuracy']:.4f} f1={stored['macro_f1']:.4f}) "
              f"{'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok or len(y) != 12000:
            problems.append(name)
    if problems:
        raise SystemExit(f"check failed for: {problems}")
    print(json.dumps({"checkpoints": len(ckpts), "out": str(args.out)}))


if __name__ == "__main__":
    main()
