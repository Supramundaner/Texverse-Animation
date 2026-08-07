"""Numerical geometry helpers shared by Blender export and unit tests."""

from __future__ import annotations

import numpy as np


def origin_centering_translation(vertices: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Return one translation whose float32 output centroid is at the origin."""
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("vertices must be a non-empty Nx3 array")
    translation = -values.mean(axis=0, dtype=np.float64)
    for _ in range(iterations):
        centered = (values + translation).astype(np.float32)
        residual = centered.mean(axis=0, dtype=np.float64)
        translation -= residual
    return translation
