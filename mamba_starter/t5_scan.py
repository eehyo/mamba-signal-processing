"""Phase 2 T5 scan transforms. Coordinates are (row, column), zero-based."""
from itertools import cycle

import numpy as np

SIZE = 14
DEFAULT_SEED = 42
SCANS = ("axis_aligned", "snake", "random", "blockwise", "spiral")
# Control condition: the outward spiral visited in reverse (edge -> centre).
# Kept out of SCANS so the main five-condition analysis is unchanged.
EXTRA_SCANS = ("spiral_inward",)


def _spiral(size):
    row = col = (size - 1) // 2
    points = [(row, col)]
    steps = 1
    for turn, (dr, dc) in enumerate(cycle(((0, 1), (1, 0), (0, -1), (-1, 0)))):
        for _ in range(steps):
            row, col = row + dr, col + dc
            if 0 <= row < size and 0 <= col < size:
                points.append((row, col))
                if len(points) == size * size:
                    return points
        steps += turn % 2


def scan_indices(scan, *, direction="raster", seed=DEFAULT_SEED):
    """Return 196 row-major flat indices in visitation order."""
    if direction not in ("raster", "column"):
        raise ValueError("direction must be raster or column")
    grid = np.arange(SIZE ** 2).reshape(SIZE, SIZE)
    if scan == "axis_aligned":
        indices = grid if direction == "raster" else grid.T
    elif scan == "snake":
        grid[1::2] = grid[1::2, ::-1].copy()
        indices = grid
    elif scan == "random":
        indices = np.random.Generator(np.random.PCG64(seed)).permutation(grid.size)
    elif scan == "blockwise":
        indices = grid.reshape(SIZE // 2, 2, SIZE // 2, 2).transpose(0, 2, 1, 3)
    elif scan == "spiral":
        row, col = np.asarray(_spiral(SIZE)).T
        indices = grid[row, col]
    elif scan == "spiral_inward":
        row, col = np.asarray(_spiral(SIZE)[::-1]).T
        indices = grid[row, col]
    else:
        raise ValueError(f"Unknown scan {scan!r}. Expected one of {SCANS + EXTRA_SCANS}")
    indices = indices.ravel()
    if not np.array_equal(np.sort(indices), np.arange(grid.size)):
        raise ValueError(f"{scan} does not visit every pixel exactly once")
    return indices


def apply_scan(images, indices):
    """Map (..., 14, 14) to (..., 196), preserving dtype and NumPy/Torch type."""
    if tuple(images.shape[-2:]) != (SIZE, SIZE):
        raise ValueError("Expected input shape (..., 14, 14)")
    return images.reshape(*images.shape[:-2], SIZE ** 2)[..., indices]
