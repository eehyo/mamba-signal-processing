"""Reproduce the frozen Part 04 evaluation. Run from the repository root."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import md5, sha256
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path("mamba_starter").resolve()))
from evaluate import evaluation_data, load_checkpoint
from train import make_loader, evaluate_model

ROOT = Path("phase2_t5")
OUT = ROOT / "results/mamba-final-test"
PROTOCOL = OUT / "PROTOCOL.md"
PROTOCOL_SHA = sha256(PROTOCOL.read_bytes()).hexdigest()
SCANS = ("raster", "snake", "random", "blockwise", "spiral")
WIDTHS = {4: "checkpoints", 1: "checkpoints-dconv1"}
SEEDS = (0, 1, 2)

def init_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@torch.inference_mode()
def probe(model, inputs):
    block = model.layers[0]
    mixer = block.mixer
    kernel = mixer.conv1d.weight[:, 0, :]
    sums = torch.zeros(2, dtype=torch.float64)
    for batch in torch.from_numpy(inputs).split(128):
        projected = mixer.in_proj(block.norm(model.input_proj(batch))).chunk(2, -1)[0]
        original = mixer.conv1d(projected.transpose(1, 2))[:, :, :196].transpose(1, 2)[:, 3:]
        substituted = projected[:, 3:] * kernel.sum(-1) + mixer.conv1d.bias
        sums[0] += (original - substituted).square().sum(dtype=torch.float64)
        sums[1] += original.square().sum(dtype=torch.float64)
    return 100 * float((sums[0] / sums[1]).sqrt())


def run_checkpoint(width, scan, seed, path, checkpoint_sha, data_root):
    model, checkpoint = load_checkpoint(path)
    config = checkpoint["config"]
    assert (checkpoint["seed"], config["scan"], config["d_conv"]) == (seed, scan, width)
    assert (config["dataset"], config["image_filter"], config["image_size"], config["image_mode"]) == ("fashion", "t5_bilinear", 14, "pixel")
    assert (config["permutation_seed"], config["batch_size"]) == (42, 64)
    inputs, labels = evaluation_data(checkpoint, str(data_root), split="test", allow_test=True)
    assert inputs.shape == (10000, 196, 1)
    assert np.array_equal(np.bincount(labels), np.full(10, 1000))
    loader = make_loader(inputs, labels, config, validation=True)
    result = evaluate_model(model, loader, "cpu", 10)
    matrix = np.array(result["confusion_matrix"])
    assert np.array_equal(matrix.sum(1), np.full(10, 1000))
    assert np.isclose(result["accuracy"], np.trace(matrix)/10000)
    assert np.isclose(result["macro_f1"], np.mean(2 * matrix.diagonal() / (matrix.sum(0) + matrix.sum(1))))
    result.update(split="test", examples=len(labels), d_conv=width, scan=scan, seed=seed,
                  checkpoint=str(path), checkpoint_sha256=checkpoint_sha, protocol_sha256=PROTOCOL_SHA,
                  relative_rms_change_pct=probe(model, inputs))
    return result


def seed_summary(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New combined results JSON path")
    parser.add_argument("--workers", default=12, type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    names = ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    md5s = ("bef4ecab320f06d8554ea6380940ec79", "bb300cfdad3c16e7a12a480ee83cd310")
    for name, expected in zip(names, md5s):
        assert md5((args.data_root / name).read_bytes()).hexdigest() == expected
    specs = []
    for width, directory in WIDTHS.items():
        manifest = json.loads((ROOT / directory / "manifest.json").read_text())
        for scan in SCANS:
            for seed in SEEDS:
                path = Path(directory) / scan / f"seed_{seed}" / "best.pt"
                expected = manifest[str(path)]["sha256"]
                assert sha256((ROOT / path).read_bytes()).hexdigest() == expected
                specs.append((width, scan, seed, ROOT / path, expected))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
        jobs = [executor.submit(run_checkpoint, *spec, args.data_root) for spec in specs]
        for job in as_completed(jobs):
            result = job.result()
            rows.append(result)
            print(f"{len(rows)}/{len(specs)} complete", flush=True)

    conditions = {}
    for width in WIDTHS:
        conditions[str(width)] = {}
        for scan in SCANS:
            runs = sorted((r for r in rows if r["d_conv"] == width and r["scan"] == scan), key=lambda r: r["seed"])
            conditions[str(width)][scan] = {
                "accuracy": seed_summary([r["accuracy"] for r in runs]),
                "macro_f1": seed_summary([r["macro_f1"] for r in runs]),
                "relative_rms_change_pct": seed_summary([r["relative_rms_change_pct"] for r in runs]),
                "seeds": runs,
            }
    output = {"split": "test", "dataset": "Fashion-MNIST", "examples": 10000, "device": "cpu",
              "torch_version": torch.__version__, "data_md5": dict(zip(names, md5s)),
              "protocol_sha256": PROTOCOL_SHA, "conditions": conditions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
