"""Lay out the published T5 checkpoints (phase2_t5/checkpoints/<scan>/seed_k/best.pt)
as run directories that predict_t5 / t5_figures / t5_delta_analysis read:

  <out>/<scan>/config.json            config stored in the checkpoint
  <out>/<scan>/seed_k/best.pt         symlink to the published checkpoint
  <out>/<scan>/seed_k/validation.json best_validation stored in the checkpoint

Each checkpoint's SHA-256 is checked against checkpoints/manifest.json, and its
best epoch / accuracy / macro-F1 against results/mamba-validation/combined_validation.json.
history.json is not published, so learning curves (F9) are unavailable for these runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

T5 = Path(__file__).resolve().parent.parent / "phase2_t5"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, default=T5 / "checkpoints")
    p.add_argument("--combined", type=Path, default=T5 / "results/mamba-validation/combined_validation.json")
    p.add_argument("--out", type=Path, default=T5 / "outputs/t5-mamba-validation")
    args = p.parse_args()

    manifest = json.loads((args.checkpoints / "manifest.json").read_text())
    combined = json.loads(args.combined.read_text())["conditions"]
    problems = []
    for ckpt in sorted(args.checkpoints.glob("*/seed_*/best.pt")):
        scan, seed_name = ckpt.parent.parent.name, ckpt.parent.name
        seed = int(seed_name.split("_")[1])
        rel = f"checkpoints/{scan}/{seed_name}/best.pt"
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        if manifest.get(rel, {}).get("sha256") != sha:
            problems.append(f"{rel}: sha256 does not match manifest")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg, best = ck["config"], ck["best_validation"]
        if cfg["scan"] != scan or ck["seed"] != seed:
            problems.append(f"{rel}: config says scan={cfg['scan']} seed={ck['seed']}")
        ref = next((r for r in combined[scan]["runs"] if r["seed"] == seed), None)
        if ref is None:
            problems.append(f"{rel}: not in combined_validation.json")
        else:
            for k in ("epoch", "accuracy", "macro_f1"):
                if abs(float(ref[k]) - float(best[k])) > 1e-9:
                    problems.append(f"{rel}: {k} {best[k]} != combined {ref[k]}")
        run = args.out / scan
        (run / seed_name).mkdir(parents=True, exist_ok=True)
        (run / "config.json").write_text(json.dumps({**cfg, "imported_from": str(args.checkpoints / scan)}, indent=2) + "\n")
        link = run / seed_name / "best.pt"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(ckpt.resolve(), link.parent))
        (run / seed_name / "validation.json").write_text(json.dumps(best, indent=2) + "\n")
        print(f"{scan} seed {seed}: epoch {best['epoch']} acc {best['accuracy']:.4f} F1 {best['macro_f1']:.4f}")
    if problems:
        raise SystemExit("problems:\n" + "\n".join(problems))
    print("all checkpoints match manifest and combined_validation.json")


if __name__ == "__main__":
    main()
