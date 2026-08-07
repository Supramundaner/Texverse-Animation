import unittest

import numpy as np

from geometry_logic import best_axis_aligned_rotation, origin_centering_translation
from pipeline_logic import CANONICAL_AXIS_ROTATIONS


class OriginCenteringTests(unittest.TestCase):
    def test_removes_float32_accumulation_residual(self) -> None:
        coordinate = np.linspace(-0.8, 0.9, 600_959, dtype=np.float32)
        vertices = np.column_stack((coordinate, coordinate * 0.7, coordinate * -0.3))
        vertices += np.array((0.17, -0.31, 0.53), dtype=np.float32)

        translation = origin_centering_translation(vertices)
        centered = (vertices.astype(np.float64) + translation).astype(np.float32)

        np.testing.assert_allclose(
            centered.mean(axis=0, dtype=np.float64), np.zeros(3), atol=1e-7
        )

    def test_rejects_empty_vertices(self) -> None:
        with self.assertRaises(ValueError):
            origin_centering_translation(np.empty((0, 3), dtype=np.float32))


class AxisAlignedGeometryRotationTests(unittest.TestCase):
    def test_recovers_axis_rotation_from_corresponding_vertices(self) -> None:
        source = np.array(
            [[-2.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 2.0, 0.25], [0.2, 0.1, 3.0]],
            dtype=np.float32,
        )
        expected_index = 9
        expected = np.asarray(CANONICAL_AXIS_ROTATIONS[expected_index][0])
        target = source @ expected.T + np.array((4.0, -3.0, 7.0))

        rotation, index, metadata = best_axis_aligned_rotation(
            source, target, [value for value, _ in CANONICAL_AXIS_ROTATIONS]
        )

        self.assertEqual(index, expected_index)
        np.testing.assert_array_equal(rotation, expected)
        self.assertLess(metadata["rms_error"], 1e-6)

    def test_rejects_mismatched_topology(self) -> None:
        with self.assertRaises(ValueError):
            best_axis_aligned_rotation(
                np.zeros((2, 3)), np.zeros((3, 3)), [np.eye(3)]
            )


if __name__ == "__main__":
    unittest.main()
