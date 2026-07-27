"""Pure, dependency-free rules shared by the TexVerse export scripts."""

from __future__ import annotations

import math
import random


PIPELINE_VERSION = "texverse-animation-v16"
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
