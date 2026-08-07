import math
import unittest

from pipeline_logic import (
    CANONICAL_AXIS_ROTATIONS,
    reference_bbox_normalization_scale,
    sampled_orbit_camera_indices,
    snap_rotation_to_axis_aligned,
    split_clips,
)


class CameraSamplingTests(unittest.TestCase):
    def test_always_selects_front_and_one_deterministic_side_camera(self) -> None:
        first = sampled_orbit_camera_indices("sample", 3)
        second = sampled_orbit_camera_indices("sample", 3)

        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        self.assertIn(first[1], range(1, 12))

    def test_requires_at_least_two_cameras(self) -> None:
        with self.assertRaises(ValueError):
            sampled_orbit_camera_indices("sample", 1, camera_count=1)


class ReferenceBBoxTests(unittest.TestCase):
    def test_uses_reference_bbox_largest_edge(self) -> None:
        self.assertEqual(
            reference_bbox_normalization_scale((-1.0, -2.0, -0.5), (1.0, 3.0, 0.5)),
            5.0,
        )

    def test_rejects_degenerate_bbox(self) -> None:
        with self.assertRaises(ValueError):
            reference_bbox_normalization_scale((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class SplitClipsTests(unittest.TestCase):
    def test_preserves_short_tail(self) -> None:
        clips, metadata = split_clips(list(range(70)), "sample", 16, 6)

        self.assertEqual([len(clip) for clip in clips], [16, 16, 16, 16, 6])
        self.assertEqual(metadata["selected_frame_count"], 70)

    def test_long_animation_uses_deterministic_contiguous_window(self) -> None:
        candidates = list(range(500))
        first, metadata = split_clips(candidates, "sample", 16, 6)
        second, _ = split_clips(candidates, "sample", 16, 6)
        flattened = [frame for clip in first for frame in clip]

        self.assertEqual(first, second)
        self.assertEqual([len(clip) for clip in first], [16] * 6)
        self.assertEqual(len(flattened), 96)
        self.assertEqual(flattened, list(range(flattened[0], flattened[0] + 96)))
        self.assertEqual(metadata["selected_frame_count"], 96)


class CanonicalRotationTests(unittest.TestCase):
    def test_has_24_unique_right_handed_rotations(self) -> None:
        matrices = [matrix for matrix, _ in CANONICAL_AXIS_ROTATIONS]
        self.assertEqual(len(matrices), 24)
        self.assertEqual(len(set(matrices)), 24)

    def test_snaps_pitch_to_axis_aligned_rotation(self) -> None:
        angle = math.radians(86)
        rotation = (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
        snapped, _, axes = snap_rotation_to_axis_aligned(rotation)
        self.assertEqual(axes, ("-Z", "+Y", "+X"))
        self.assertEqual(snapped, ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))

    def test_snaps_combined_3d_rotation(self) -> None:
        expected, expected_axes = CANONICAL_AXIS_ROTATIONS[7]
        noisy = tuple(
            tuple(value + (0.02 if row == column else -0.01) for column, value in enumerate(row_values))
            for row, row_values in enumerate(expected)
        )
        snapped, index, axes = snap_rotation_to_axis_aligned(noisy)
        self.assertEqual(index, 7)
        self.assertEqual(snapped, expected)
        self.assertEqual(axes, expected_axes)


if __name__ == "__main__":
    unittest.main()
