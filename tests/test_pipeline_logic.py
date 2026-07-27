import math
import unittest

from pipeline_logic import snap_yaw_to_quarter_turn, split_clips, wrap_angle


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


class CanonicalYawTests(unittest.TestCase):
    def test_snaps_to_nearest_quarter_turn(self) -> None:
        cases = [
            (math.radians(10), 0.0, 0),
            (math.radians(50), math.pi / 2.0, 1),
            (math.radians(140), -math.pi, 2),
            (math.radians(-80), -math.pi / 2.0, 3),
        ]
        for angle, expected, expected_index in cases:
            with self.subTest(angle=angle):
                snapped, index = snap_yaw_to_quarter_turn(angle)
                self.assertAlmostEqual(snapped, expected)
                self.assertEqual(index, expected_index)

    def test_wrapping_is_stable(self) -> None:
        self.assertAlmostEqual(wrap_angle(2.0 * math.pi), 0.0)
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi), -math.pi)


if __name__ == "__main__":
    unittest.main()
