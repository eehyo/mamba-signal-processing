"""Train on a fixed development partition. This CLI never reads official test data."""
import argparse
import json
from pathlib import Path
import random
import platform
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data import development_data, Standardizer, SequenceDataset, collate_sequences
from models import build_model
from signal_utils import downsample


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def metrics(y, pred, num_classes):
    labels = np.arange(num_classes)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist()}


def make_loader(x, y, cfg, shuffle=False, seed=0, validation=False):
    ds = SequenceDataset(x, y, cfg["keep_rate"], cfg["gap_mode"],
                         cfg["missing_seed"] + (1000000 if validation else 0))
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=0,
                      collate_fn=collate_sequences, generator=torch.Generator().manual_seed(seed))


def evaluate_model(model, loader, device, num_classes):
    model.eval()
    ys, preds, lengths = [], [], []
    with torch.no_grad():
        for batch in loader:
            dt = batch["dt"].to(device) if batch["dt"] is not None else None
            logits = model(batch["x"].to(device), dt_mult=dt, mask=batch["mask"].to(device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite evaluation logits")
            ys.extend(batch["y"].tolist())
            preds.extend(logits.argmax(-1).cpu().tolist())
            lengths.extend(batch["mask"].sum(1).tolist())
    result = metrics(ys, preds, num_classes)
    result.update(valid_length_min=int(min(lengths)), valid_length_max=int(max(lengths)),
                  valid_length_mean=float(np.mean(lengths)))
    return result


def model_kwargs(cfg, input_dim, num_classes):
    return dict(name=cfg["model"], input_dim=input_dim, num_classes=num_classes,
                d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_state=cfg["d_state"],
                selective=not cfg["lti"], dt_min=cfg["dt_min"], dt_max=cfg["dt_max"], d_conv=cfg["d_conv"])


def train_seed(cfg, seed, xtr, ytr, xva, yva, norm, manifest):
    seed_everything(seed)
    device = torch.device(cfg["device"])
    num_classes = 10 if cfg["dataset"] == "fashion" else 6
    kwargs = model_kwargs(cfg, xtr.shape[-1] + (cfg["gap_mode"] == "channel"), num_classes)
    model = build_model(**kwargs).to(device)
    if cfg["model"] == "mamba":
        model.set_delta_scale(cfg["delta_scale"])
    groups = {True: [], False: []}
    for name, p in model.named_parameters():
        # Preserve decay/skip initialization; biases/norms also have no decay.
        decay = p.ndim > 1 and not name.endswith("A_log")
        groups[decay].append(p)
    optimizer = torch.optim.AdamW([{"params": groups[True], "weight_decay": cfg["weight_decay"]},
                                   {"params": groups[False], "weight_decay": 0.0}], lr=cfg["lr"])
    train_loader = make_loader(xtr, ytr, cfg, shuffle=True, seed=seed)
    val_loader = make_loader(xva, yva, cfg, validation=True)
    best_score, best_metrics, history = -1.0, None, []
    out = Path(cfg["out"]) / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            x, y, mask = batch["x"].to(device), batch["y"].to(device), batch["mask"].to(device)
            dt = batch["dt"].to(device) if batch["dt"] is not None else None
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x, dt_mult=dt, mask=mask), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        val = evaluate_model(model, val_loader, device, num_classes)
        history.append({"epoch": epoch, "train_loss": total/count, **val})
        print(f"seed={seed} epoch={epoch} loss={total/count:.4f} val_acc={val['accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f}", flush=True)
        if val["macro_f1"] > best_score:
            best_score, best_metrics = val["macro_f1"], {"epoch": epoch, **val}
            torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "model_kwargs": kwargs, "config": cfg, "normalizer": norm.state_dict(),
                        "split": manifest, "seed": seed, "best_validation": best_metrics}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "validation.json").write_text(json.dumps(best_metrics, indent=2) + "\n")
    return {"seed": seed, "parameters": sum(p.numel() for p in model.parameters()), **best_metrics}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["har", "fashion", "synthetic"], default="har")
    p.add_argument("--data_root", default="data/har")
    p.add_argument("--download", action="store_true")
    p.add_argument("--model", choices=["mamba", "cnn", "lstm"], default="mamba")
    p.add_argument("--lti", action="store_true")
    p.add_argument("--out", default="runs/mamba")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    for key, default in [("epochs",20),("batch_size",64),("d_model",32),("d_state",16),
                         ("n_layers",2),("d_conv",4),("split_seed",42),("subset_seed",123),
                         ("permutation_seed",42),("missing_seed",456),("image_size",14),("threads",2)]:
        p.add_argument("--"+key, type=int, default=default)
    p.add_argument("--train_subset", type=int)
    p.add_argument("--val_subjects", type=int, nargs="+")
    for key, default in [("lr",1e-3),("weight_decay",0.01),("val_fraction",0.2),
                         ("dt_min",0.001),("dt_max",0.1),("delta_scale",1.0),("keep_rate",1.0)]:
        p.add_argument("--"+key, type=float, default=default)
    p.add_argument("--downsample_factor", type=int, choices=[1,2,4,8], default=1)
    p.add_argument("--downsample_method", choices=["aa", "naive"], default="aa")
    p.add_argument("--gap_mode", choices=["ignore","interpolate","delta","channel"], default="ignore")
    p.add_argument("--image_filter", choices=["blur","average","naive"], default="blur")
    p.add_argument("--image_mode", choices=["pixel","row"], default="pixel")
    p.add_argument("--scan", choices=["raster","snake","column","random"], default="raster")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main():
    cfg = vars(parser().parse_args())
    if min(cfg[k] for k in ["epochs","batch_size","d_model","d_state","n_layers","d_conv","threads"]) < 1:
        raise ValueError("training counts and model dimensions must be positive")
    if len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be distinct")
    if cfg["model"] != "mamba" and (cfg["lti"] or cfg["delta_scale"] != 1 or cfg["gap_mode"] == "delta"):
        raise ValueError("lti and delta interventions require --model mamba")
    if cfg["dataset"] == "fashion" and (cfg["downsample_factor"] != 1 or cfg["keep_rate"] != 1):
        raise ValueError("use image_size/image_filter for image experiments")
    torch.set_num_threads(cfg["threads"])
    out = Path(cfg["out"])
    if (out / "config.json").exists():
        raise FileExistsError("output already contains a run; choose a new --out")
    xtr, ytr, xva, yva, manifest = development_data(cfg)
    expected = set(range(10 if cfg["dataset"] == "fashion" else 6))
    if set(ytr.tolist()) != expected or set(yva.tolist()) != expected:
        raise ValueError("every class must be represented in train and validation; choose another subject split/subset")
    norm = Standardizer().fit(xtr)  # Fit ONLY on the actual training subset.
    xtr, xva = norm.transform(xtr), norm.transform(xva)
    if cfg["downsample_factor"] > 1:
        xtr = downsample(xtr, cfg["downsample_factor"], cfg["downsample_method"])
        xva = downsample(xva, cfg["downsample_factor"], cfg["downsample_method"])
    if xtr.shape[1] > 200:
        raise ValueError("sequence exceeds 200 steps; downsample/window first (e.g. 14x14 pixel mode)")
    out.mkdir(parents=True, exist_ok=True)
    cfg["versions"] = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = [train_seed(cfg, seed, xtr, ytr, xva, yva, norm, manifest) for seed in cfg["seeds"]]
    summary = {"split": "validation", "std_convention": "sample std, ddof=1; null for one seed", "runs": results}
    for metric in ["accuracy", "macro_f1"]:
        values = [r[metric] for r in results]
        summary[metric] = {"mean": float(np.mean(values)),
                           "std": float(np.std(values, ddof=1)) if len(values)>1 else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["accuracy", "macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
