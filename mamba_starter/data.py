"""Raw UCI HAR inertial windows, Fashion-MNIST IDX files, and padding utilities."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import struct
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from signal_utils import irregular_sample, interpolate_irregular
from image_utils import resize_images, image_to_sequence

HAR_CHANNELS = [f"{signal}_{axis}" for signal in ("body_acc", "body_gyro", "total_acc")
                for axis in "xyz"]
HAR_CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
HAR_URL = "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def _download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as src, part.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def _safe_extract(archive, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (root / info.filename).resolve()
            if not target.is_relative_to(root) or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("unsafe zip member")
        z.extractall(root)


def download_har(root):
    root = Path(root)
    archive = root / "uci_har.zip"
    _download(HAR_URL, archive)
    _safe_extract(archive, root)
    nested = root / "UCI HAR Dataset.zip"
    if nested.exists():
        _safe_extract(nested, root)
    (root / "download_manifest.json").write_text(json.dumps({"url": HAR_URL,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2) + "\n")


def load_har(root, split="train", *, allow_test=False, download=False):
    """Never read the official test split unless explicitly requested for final reporting."""
    if split not in {"train", "test"}:
        raise ValueError("HAR split must be train or test")
    if split == "test" and not allow_test:
        raise ValueError("official test is reserved for a fixed final protocol; set allow_test=True")
    root = Path(root)
    if not (root / "train").is_dir():
        root = root / "UCI HAR Dataset"
    if not root.is_dir() and download:
        download_har(root.parent)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing UCI HAR Dataset under {root}; use --download or extract the official ZIP")
    x = np.stack([np.loadtxt(root / split / "Inertial Signals" / f"{c}_{split}.txt",
                            dtype=np.float32, ndmin=2) for c in HAR_CHANNELS], axis=-1)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64, ndmin=1) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64, ndmin=1)
    if x.shape != (len(y), 128, 9) or len(subjects) != len(y) or not np.isfinite(x).all():
        raise ValueError("invalid HAR shapes or non-finite signals")
    if not np.isin(y, np.arange(6)).all():
        raise ValueError("HAR labels must be original values 1..6 (converted to 0..5)")
    return x, y, subjects


def subject_split(subjects, val_fraction=0.2, split_seed=42, val_subjects=None):
    """A fixed subject partition independent of model/training seed."""
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    if len(unique) < 2 or not 0 < val_fraction < 1:
        raise ValueError("need at least two subjects and 0 < val_fraction < 1")
    if val_subjects is None:
        n = min(len(unique)-1, max(1, int(np.ceil(val_fraction * len(unique)))))
        val_subjects = np.random.default_rng(split_seed).permutation(unique)[:n]
    val_subjects = np.asarray(val_subjects)
    if (not len(val_subjects) or not np.isin(val_subjects, unique).all()
            or len(np.unique(val_subjects)) >= len(unique)):
        raise ValueError("validation subjects must be a nonempty proper subset")
    is_val = np.isin(subjects, val_subjects)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def subset_indices(y, n=None, seed=123):
    """Fixed stratified training subset, including near-full and rare-class cases.

    The unused complement does not need every class. Preserve the original
    scikit-learn split for ordinary cases so existing experiment seeds are stable.
    """
    y = np.asarray(y)
    if y.ndim != 1 or not len(y):
        raise ValueError("y must be a nonempty one-dimensional label array")
    if n is not None and (not isinstance(n, (int, np.integer)) or n < 1):
        raise ValueError("train_subset must be a positive integer")
    if n is None or n >= len(y):
        return np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    if n < len(classes):
        raise ValueError("train_subset must include at least one example per class")
    if counts.min() >= 2 and len(y) - n >= len(classes):
        chosen, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
        if len(np.unique(y[chosen])) == len(classes):
            return np.sort(chosen)
    # Constrained proportional allocation: at least one per class, at most the
    # available count; adjust largest rounding deficits/excesses to total n.
    target = n * counts / len(y)
    allocation = np.clip(np.floor(target).astype(int), 1, counts)
    while allocation.sum() < n:
        deficits = np.where(allocation < counts, target - allocation, -np.inf)
        allocation[np.argmax(deficits)] += 1
    while allocation.sum() > n:
        excesses = np.where(allocation > 1, allocation - target, -np.inf)
        allocation[np.argmax(excesses)] -= 1
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(np.flatnonzero(y == label), count, replace=False)
                             for label, count in zip(classes, allocation)])
    return np.sort(chosen)


class Standardizer:
    def fit(self, x):
        self.mean = np.asarray(x, dtype=np.float64).mean((0, 1), keepdims=True)
        self.std = np.maximum(np.asarray(x, dtype=np.float64).std((0, 1), keepdims=True), 1e-6)
        return self

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.mean, obj.std = np.asarray(state["mean"]), np.asarray(state["std"])
        return obj


def _idx(path, image):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != (2051 if image else 2049):
            raise ValueError(f"bad IDX magic in {path}")
        if image:
            h, w = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), np.uint8).reshape(n, h, w).copy()
        return np.frombuffer(f.read(), np.uint8).reshape(n).astype(np.int64)


def load_fashion(root, split="train", *, allow_test=False, download=False):
    if split not in {"train", "test"} or (split == "test" and not allow_test):
        raise ValueError("use train for development; official test requires allow_test=True")
    prefix = "train" if split == "train" else "t10k"
    paths = [Path(root) / f"{prefix}-{suffix}.gz" for suffix in ("images-idx3-ubyte", "labels-idx1-ubyte")]
    for p in paths:
        if download:
            _download(FASHION_URL + p.name, p)
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}; use --download or place official IDX gzip files here")
    images, labels = _idx(paths[0], True), _idx(paths[1], False)
    if len(images) != len(labels) or images.shape[1:] != (28, 28) or not np.isin(labels, range(10)).all():
        raise ValueError("invalid Fashion-MNIST data")
    return images, labels


def image_sequences(images, cfg, size=None, method=None):
    # Two-stage 28->training size->test size isolates the T6 intervention.
    x = resize_images(images.astype(np.float32) / 255, cfg["image_size"], cfg["image_filter"])
    if size is not None and size != cfg["image_size"]:
        x = resize_images(x, size, method or cfg["image_filter"])
    return image_to_sequence(x, cfg["scan"], cfg["image_mode"], cfg["permutation_seed"])


def synthetic_data(n=240, length=64, seed=314):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 6
    t = np.arange(length) / 50
    phase = rng.uniform(0, 2*np.pi, (n, 1, 1))
    x = np.sin(2*np.pi*(1 + 2*y[:, None, None])*t[None, :, None] + phase)
    x = x * np.array([1, 0.7, 0.4])[None, None, :] + rng.normal(0, 0.1, (n, length, 3))
    return x.astype(np.float32), y.astype(np.int64)


def development_data(cfg):
    """Read training files ONLY; return train, val, fixed split manifest."""
    if cfg["dataset"] == "har":
        x, y, subjects = load_har(cfg["data_root"], download=cfg["download"])
        tr, va = subject_split(subjects, cfg["val_fraction"], cfg["split_seed"], cfg.get("val_subjects"))
        manifest = {"train_subjects": np.unique(subjects[tr]).tolist(),
                    "val_subjects": np.unique(subjects[va]).tolist(), "split_seed": cfg["split_seed"]}
    elif cfg["dataset"] == "fashion":
        images, y = load_fashion(cfg["data_root"], download=cfg["download"])
        tr, va = train_test_split(np.arange(len(y)), test_size=cfg["val_fraction"],
                                 random_state=cfg["split_seed"], stratify=y)
        # Select images before sequence conversion to keep memory modest.
        tr = tr[subset_indices(y[tr], cfg["train_subset"], cfg["subset_seed"])]
        manifest = {"split_seed": cfg["split_seed"], "train_indices": tr.tolist(), "val_indices": va.tolist()}
        return image_sequences(images[tr], cfg), y[tr], image_sequences(images[va], cfg), y[va], manifest
    else:
        x, y = synthetic_data(seed=cfg["split_seed"])
        tr, va = train_test_split(np.arange(len(y)), test_size=cfg["val_fraction"],
                                 random_state=cfg["split_seed"], stratify=y)
        manifest = {"split_seed": cfg["split_seed"], "synthetic_only": True}
    tr = tr[subset_indices(y[tr], cfg["train_subset"], cfg["subset_seed"])]
    manifest.update(train_indices=tr.tolist(), val_indices=va.tolist())
    return x[tr], y[tr], x[va], y[va], manifest


class SequenceDataset(Dataset):
    def __init__(self, x, y, keep_rate=1.0, gap_mode="ignore", missing_seed=456):
        if len(x) != len(y) or not len(x):
            raise ValueError("nonempty x and y must have the same length")
        if gap_mode not in {"ignore", "interpolate", "delta", "channel"} or not 0 < keep_rate <= 1:
            raise ValueError("invalid gap strategy or keep rate")
        self.x, self.y = x, y
        self.keep_rate, self.gap_mode, self.missing_seed = keep_rate, gap_mode, missing_seed

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        gaps = np.ones(len(x), dtype=np.float32)
        if self.keep_rate < 1:
            x, ticks, gaps = irregular_sample(x, self.keep_rate, np.random.default_rng(self.missing_seed + i))
            if self.gap_mode == "interpolate":
                x = interpolate_irregular(x, ticks, len(self.x[i]))
                gaps = np.ones(len(x), dtype=np.float32)
        if self.gap_mode == "channel":
            x = np.concatenate([x, gaps[:, None]], axis=-1)
        return {"x": torch.tensor(x, dtype=torch.float32), "y": int(self.y[i]),
                "dt": torch.tensor(gaps) if self.gap_mode == "delta" else None}


def collate_sequences(batch):
    lengths = torch.tensor([len(item["x"]) for item in batch])
    x = torch.nn.utils.rnn.pad_sequence([v["x"] for v in batch], batch_first=True)
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    flags = [v["dt"] is not None for v in batch]
    if any(flags) and not all(flags):
        raise ValueError("mixed dt strategies in one batch")
    dt = torch.nn.utils.rnn.pad_sequence([v["dt"] for v in batch], batch_first=True) if all(flags) else None
    return {"x": x, "y": torch.tensor([v["y"] for v in batch]), "mask": mask, "dt": dt}
