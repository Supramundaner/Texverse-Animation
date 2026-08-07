import argparse
import tempfile
import unittest
from pathlib import Path

from process_texverse_animation import processing_config, publish_processed_root


class PublishProcessedRootTests(unittest.TestCase):
    def test_replaces_old_tree_after_new_tree_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged = root / "staged"
            final = root / "final"
            staged.mkdir()
            final.mkdir()
            (staged / "new.txt").write_text("new")
            (final / "old.txt").write_text("old")

            publish_processed_root(staged, final)

            self.assertEqual((final / "new.txt").read_text(), "new")
            self.assertFalse((final / "old.txt").exists())
            self.assertFalse(final.with_name(".final.previous").exists())


class ProcessingConfigTests(unittest.TestCase):
    def test_contains_every_output_affecting_option(self) -> None:
        args = argparse.Namespace(
            resolution=512,
            clip_frames=16,
            max_clips=6,
            max_fps=16.0,
            min_motion=0.01,
            max_centroid_motion_bbox_ratio=1.0,
            min_image_change=0.01,
            image_change_pixel_threshold=10,
            min_reference_foreground=0.05,
            reference_background_pixel_threshold=10,
            render_threads=12,
        )

        self.assertEqual(set(processing_config(args)), {
            "resolution",
            "clip_frames",
            "max_clips",
            "max_fps",
            "min_motion",
            "max_centroid_motion_bbox_ratio",
            "min_image_change",
            "image_change_pixel_threshold",
            "min_reference_foreground",
            "reference_background_pixel_threshold",
            "render_threads",
        })


if __name__ == "__main__":
    unittest.main()
