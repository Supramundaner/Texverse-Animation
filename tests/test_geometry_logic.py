import unittest

import numpy as np

from geometry_logic import origin_centering_translation


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


if __name__ == "__main__":
    unittest.main()
