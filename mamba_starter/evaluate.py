"""Evaluate a saved checkpoint under one explicit intervention, on validation by default."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data import load_har, load_fashion, image_sequences, synthetic_data, Standardizer
from models import build_model
from signal_utils import downsample, upsample
from train import make_loader, evaluate_model


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def evaluation_data(ckpt, data_root=None, split="validation", allow_test=False,
                    image_size=None, image_filter=None, download=False):
    cfg = ckpt["config"]
    root = data_root or cfg["data_root"]
    if split not in {"validation", "test"} or (split == "test" and not allow_test):
        raise ValueError("test requires an explicit fixed protocol")
    if cfg["dataset"] == "har":
        x, y, _ = load_har(root, "test" if split == "test" else "train", allow_test=allow_test, download=download)
    elif cfg["dataset"] == "fashion":
        images, y = load_fashion(root, "test" if split == "test" else "train", allow_test=allow_test, download=download)
        if split == "validation":
            idx = ckpt["split"]["val_indices"]
            images, y = images[idx], y[idx]
        x = image_sequences(images, cfg, image_size, image_filter)
        return Standardizer.from_state(ckpt["normalizer"]).transform(x), y
    else:
        if split == "test":
            raise ValueError("synthetic smoke data has no official test split")
        x, y = synthetic_data(seed=cfg["split_seed"])
    if split == "validation":
        idx = ckpt["split"]["val_indices"]
        x, y = x[idx], y[idx]
    return Standardizer.from_state(ckpt["normalizer"]).transform(x), y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root")
    p.add_argument("--download", action="store_true")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--final_protocol", help="Path to the frozen protocol document; required for --split test")
    p.add_argument("--out", required=True)
    p.add_argument("--factor", type=int, choices=[1,2,4,8])
    p.add_argument("--method", choices=["naive", "aa"], help="Defaults to the checkpoint's training method")
    p.add_argument("--upsample", choices=["linear", "poly"])
    p.add_argument("--delta_scale", type=float)
    p.add_argument("--keep_rate", type=float)
    p.add_argument("--gap_mode", choices=["ignore","interpolate","delta","channel"])
    p.add_argument("--image_size", type=int)
    p.add_argument("--image_filter", choices=["blur","average","naive"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    protocol_sha256 = None
    if args.split == "test":
        if not args.final_protocol or not Path(args.final_protocol).is_file():
            p.error("--split test requires --final_protocol PATH, fixed before any final evaluation")
        protocol_bytes = Path(args.final_protocol).read_bytes()
        if not protocol_bytes.strip():
            p.error("the fixed protocol file must not be empty")
        protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    cfg = ckpt["config"].copy()
    factor = args.factor if args.factor is not None else cfg["downsample_factor"]
    method = args.method or cfg["downsample_method"]
    if cfg["dataset"] == "fashion":
        if factor != 1 or args.upsample:
            p.error("image experiments use --image_size and --image_filter")
        if cfg["image_mode"] == "row" and args.image_size not in (None, cfg["image_size"]):
            p.error("row-mode width is input_dim; resolution transfer requires pixel mode")
    elif args.image_size is not None:
        p.error("image_size is only for Fashion-MNIST")
    x, y = evaluation_data(ckpt, args.data_root, args.split, args.split == "test",
                           args.image_size, args.image_filter, args.download)
    if cfg["dataset"] != "fashion":
        x = downsample(x, factor, method)
        if args.upsample:
            train_factor = cfg["downsample_factor"]
            if factor % train_factor:
                p.error("upsampling requires an integer factor back to the training rate")
            x = upsample(x, factor // train_factor, args.upsample)
    for key in ["keep_rate", "gap_mode"]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if cfg["dataset"] == "fashion" and cfg["keep_rate"] != 1:
        p.error("irregular-time simulator is for time series")
    if x.shape[-1] + (cfg["gap_mode"] == "channel") != ckpt["model_kwargs"]["input_dim"]:
        p.error("gap-channel evaluation requires a model trained with --gap_mode channel")
    if cfg["model"] != "mamba" and (args.delta_scale is not None or cfg["gap_mode"] == "delta"):
        p.error("delta interventions require Mamba")
    if args.delta_scale is not None:
        model.set_delta_scale(args.delta_scale)
    loader = make_loader(x, y, cfg, validation=True)
    result = evaluate_model(model, loader, args.device, ckpt["model_kwargs"]["num_classes"])
    reference_rate = (50/(cfg["downsample_factor"] if args.upsample else factor)
                      if cfg["dataset"] == "har" else None)
    irregular_grid = cfg["keep_rate"] < 1 and cfg["gap_mode"] != "interpolate"
    result.update(split=args.split, seed=ckpt["seed"], checkpoint=str(args.checkpoint),
                  protocol_sha256=protocol_sha256, intervention=vars(args), examples=len(y),
                  input_length=x.shape[1], gap_mode=cfg["gap_mode"], keep_rate=cfg["keep_rate"],
                  input_length_convention="regular-grid length before dropping; see valid_length_* for observed tokens",
                  resolved_factor=factor, resolved_downsample_method=method,
                  actual_delta_scale=[float(layer.mixer.delta_scale) for layer in model.layers]
                      if cfg["model"] == "mamba" else None,
                  reference_grid_rate_hz=reference_rate, irregular_grid=irregular_grid,
                  sample_rate_hz=None if irregular_grid else reference_rate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
