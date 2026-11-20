"""Fashion-MNIST IDX loading and shared antialiased resizing."""
import gzip
import struct

import numpy as np
from scipy.ndimage import convolve1d


def read_images(root, split):
    prefix = {"train": "train", "test": "t10k"}[split]
    raw = root / f"{prefix}-images-idx3-ubyte"
    opener, path = (open, raw) if raw.exists() else (gzip.open, raw.with_suffix(".gz"))
    with opener(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        pixels = np.frombuffer(file.read(), dtype=np.uint8)
    if (magic, rows, cols) != (2051, 28, 28) or pixels.size != count * rows * cols:
        raise ValueError(f"Invalid Fashion-MNIST image IDX file: {path}")
    return pixels.reshape(count, rows, cols)


def downsample(images):
    """Antialiased 28×28 to 14×14 bilinear downsampling of uint8 pixels."""
    if images.shape[-2:] != (28, 28) or images.dtype != np.uint8:
        raise ValueError("Expected uint8 images with shape (..., 28, 28)")
    pixels = images.astype(np.float32) / 255
    # Widen the triangular filter for scale 2 and sample at half-pixel centers.
    weights = np.array([1, 3, 3, 1], dtype=np.float32) / 8
    for axis in (-2, -1):
        pixels = convolve1d(pixels, weights, axis=axis, mode="constant")
        valid_weight = convolve1d(np.ones(28, dtype=np.float32), weights,
                                  mode="constant")
        shape = [1] * pixels.ndim
        shape[axis] = 28
        pixels /= valid_weight.reshape(shape)
    return pixels[..., ::2, ::2].copy()
