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


def best_axis_aligned_rotation(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    rotations,
) -> tuple[np.ndarray, int, dict]:
    """Choose the axis rotation minimizing centered point-correspondence error."""
    source = np.asarray(source_vertices, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) == 0:
        raise ValueError("source_vertices must be a non-empty Nx3 array")
    if target.shape != source.shape:
        raise ValueError("target_vertices must have the same Nx3 shape as source_vertices")
    candidates = [np.asarray(rotation, dtype=np.float64) for rotation in rotations]
    if not candidates or any(rotation.shape != (3, 3) for rotation in candidates):
        raise ValueError("rotations must contain at least one 3x3 matrix")

    source_centered = source - source.mean(axis=0, dtype=np.float64)
    target_centered = target - target.mean(axis=0, dtype=np.float64)
    covariance = target_centered.T @ source_centered
    scores = [float(np.sum(rotation * covariance)) for rotation in candidates]
    best_index = int(np.argmax(scores))
    rotation = candidates[best_index]
    aligned = source_centered @ rotation.T
    rms_error = float(np.sqrt(np.mean(np.sum((aligned - target_centered) ** 2, axis=1))))
    target_rms_radius = float(
        np.sqrt(np.mean(np.sum(target_centered * target_centered, axis=1)))
    )
    return rotation, best_index, {
        "method": "minimum_centered_corresponding_vertex_rms_over_axis_rotations",
        "vertex_count": int(len(source)),
        "rms_error": rms_error,
        "normalized_rms_error": rms_error / max(target_rms_radius, 1e-12),
        "score": scores[best_index],
    }
