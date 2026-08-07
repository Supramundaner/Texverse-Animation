"""Pure, dependency-free rules shared by the TexVerse export scripts."""

from __future__ import annotations

import math
import random
import hashlib


PIPELINE_VERSION = "texverse-animation-v26"
QUARTER_TURN = math.pi / 2.0


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def snap_yaw_to_quarter_turn(angle: float) -> tuple[float, int]:
    """Return the nearest world-axis yaw and its quarter-turn index."""
    index = math.floor(angle / QUARTER_TURN + 0.5)
    snapped = wrap_angle(index * QUARTER_TURN)
    canonical_index = int(round(snapped / QUARTER_TURN)) % 4
    return snapped, canonical_index


def sampled_orbit_camera_indices(
    sample_id: str,
    clip_index: int,
    camera_count: int = 12,
) -> list[int]:
    """Select camera_0 plus one deterministic non-front orbit camera."""
    if camera_count < 2:
        raise ValueError("camera_count must be at least 2")
    digest = hashlib.sha256(f"{sample_id}:{clip_index}:camera".encode()).hexdigest()
    return [0, 1 + (int(digest[:16], 16) % (camera_count - 1))]


def reference_bbox_normalization_scale(
    minimum: Sequence[float],
    maximum: Sequence[float],
) -> float:
    """Return the reference mesh bbox's largest edge for motion normalization."""
    if len(minimum) != 3 or len(maximum) != 3:
        raise ValueError("reference bbox bounds must contain three coordinates")
    extents = [float(high) - float(low) for low, high in zip(minimum, maximum)]
    if not all(math.isfinite(value) for value in extents):
        raise ValueError("reference bbox bounds must be finite")
    if any(value < 0.0 for value in extents):
        raise ValueError("reference bbox minimum must not exceed maximum")
    scale = max(extents)
    if scale <= 0.0:
        raise ValueError("reference bbox must have a positive extent")
    return scale


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
