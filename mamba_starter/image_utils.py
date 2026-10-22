"""Grayscale [batch,height,width] -> sequence. Frequencies are cycles/token."""
import numpy as np
from scipy.ndimage import gaussian_filter


def resize_images(images, size=14, method="blur"):
    x = np.asarray(images, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != x.shape[2] or size < 1 or x.shape[1] % size:
        raise ValueError("expect square images and an integer downsampling factor")
    m = x.shape[1] // size
    if method not in {"blur", "average", "naive"}:
        raise ValueError("method must be blur, average, or naive")
    if m == 1:
        return x.copy()
    if method == "blur":
        # Reflect boundary, Gaussian sigma=m/2 pixels, truncate at 4 sigma.
        x = gaussian_filter(x, sigma=(0, m/2, m/2), mode="reflect", truncate=4.0)
    elif method == "average":
        return x.reshape(len(x), size, m, size, m).mean((2, 4))
    return x[:, ::m, ::m].copy()


def image_to_sequence(images, scan="raster", mode="pixel", permutation_seed=42):
    x = np.asarray(images, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("expect grayscale images [batch,height,width]")
    if mode == "row":
        if scan != "raster":
            raise ValueError("row mode uses one row per token; choose scan=raster")
        return x.copy()  # [batch,height,width], width is input_dim
    if mode != "pixel":
        raise ValueError("mode must be pixel or row")
    if scan == "snake":
        x = x.copy()
        x[:, 1::2] = x[:, 1::2, ::-1]
    elif scan == "column":
        x = x.transpose(0, 2, 1)
    elif scan not in {"raster", "random"}:
        raise ValueError("scan must be raster, snake, column, or random")
    seq = x.reshape(len(x), -1, 1)
    if scan == "random":
        # Same permutation for all images/splits/model seeds at a fixed resolution.
        p = np.random.default_rng(permutation_seed).permutation(seq.shape[1])
        seq = seq[:, p]
    return seq.copy()
