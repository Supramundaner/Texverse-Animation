"""Pure, dependency-free rules shared by the TexVerse export scripts."""

from __future__ import annotations

import math
import random
from itertools import permutations, product
from typing import Sequence


PIPELINE_VERSION = "texverse-animation-v17"


def _permutation_sign(permutation: tuple[int, int, int]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


def canonical_axis_rotations() -> list[tuple[tuple[tuple[float, ...], ...], tuple[str, ...]]]:
    """Return the 24 right-handed rotations whose axes align to world axes."""
    axis_names = ("X", "Y", "Z")
    rotations = []
    for permutation in permutations(range(3)):
        permutation_sign = _permutation_sign(permutation)
        for signs in product((-1, 1), repeat=3):
            if permutation_sign * math.prod(signs) != 1:
                continue
            rows = [[0.0] * 3 for _ in range(3)]
            labels = []
            for local_axis, (world_axis, sign) in enumerate(zip(permutation, signs)):
                rows[world_axis][local_axis] = float(sign)
                labels.append(("+" if sign > 0 else "-") + axis_names[world_axis])
            rotations.append((tuple(tuple(row) for row in rows), tuple(labels)))
    return rotations


CANONICAL_AXIS_ROTATIONS = canonical_axis_rotations()


def snap_rotation_to_axis_aligned(
    rotation: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[float, ...], ...], int, tuple[str, ...]]:
    """Snap a 3D rotation to the nearest of the 24 axis-aligned rotations."""
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError("rotation must be a 3x3 matrix")
    values = tuple(tuple(float(value) for value in row) for row in rotation)
    # For rotation matrices, minimizing Frobenius distance is equivalent to
    # maximizing their element-wise inner product.
    best_index = max(
        range(len(CANONICAL_AXIS_ROTATIONS)),
        key=lambda index: sum(
            CANONICAL_AXIS_ROTATIONS[index][0][row][column] * values[row][column]
            for row in range(3)
            for column in range(3)
        ),
    )
    snapped, axes = CANONICAL_AXIS_ROTATIONS[best_index]
    return snapped, best_index, axes


def split_clips(
    candidates: list[int],
    sample_id: str,
    clip_frames: int,
    max_clips: int,
) -> tuple[list[list[int]], dict]:
    """Select one deterministic window and split it into bounded clips."""
    if clip_frames < 1 or max_clips < 1:
        raise ValueError("clip_frames and max_clips must be positive")

    capacity = clip_frames * max_clips
    if len(candidates) > capacity:
        start = random.Random(sample_id).randint(0, len(candidates) - capacity)
        selected = candidates[start : start + capacity]
        method = "deterministic_contiguous_random_window_limited_to_clip_capacity"
    else:
        selected = candidates
        method = "all_effective_frames"

    clips = [selected[index : index + clip_frames] for index in range(0, len(selected), clip_frames)]
    return clips, {
        "method": method,
        "effective_frame_count": len(candidates),
        "selected_frame_count": len(selected),
        "clip_frames": clip_frames,
        "max_clips": max_clips,
        "clip_count": len(clips),
    }
