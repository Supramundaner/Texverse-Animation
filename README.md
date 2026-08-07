# Texverse Animation

Blender-based preprocessing for TexVerse FBX/GLB animations. The pipeline
normalizes each asset, measures translation-invariant motion, splits animations
into independent clips, and exports reference meshes, animated vertices,
renders, cameras, and manifests.

## Output rules

- Downsample source animation to at most 16 FPS.
- Select at most one deterministic contiguous 96-frame window.
- Split the selected window into at most six clips of at most 16 frames.
- Keep a clip only when de-centered sampled-vertex motion, normalized by the
  reference mesh bounding box's largest edge, is greater than 0.01.
- Keep a clip only when its maximum centroid excursion from the first target
  frame does not exceed one reference-pose bounding-box edge.
- After rendering, keep a clip only when consecutive target frames change at
  least 1.0% of pixels on average, using a maximum RGB-channel delta of 10/255.
- Before rendering target frames, keep a clip only when at least 5% of its
  reference pixels differ from the median border background by more than 10/255.
- Compute reference alignment, camera bounds, validity, and renders independently
  for every clip.
- Fit the fixed clip camera using Blender's effective square-frame projection,
  with five percent safety margin per limiting image edge.
- For rigged assets, align REST-pose root scale to the first animation frame and
  align its complete 3D orientation to the nearest of the 24 right-handed,
  axis-aligned 90-degree canonical rotations of that first frame.
- For unrigged assets, restore the original imported reference frame before
  applying per-clip alignment.

## Files

- `export_texverse_animation.py`: Blender worker for one asset.
- `process_texverse_animation.py`: resumable multi-process supervisor.
- `pipeline_logic.py`: dependency-free clip and canonical-orientation rules.
- `run_texverse_animation_batch.sh`: default four-worker launcher.
- `tests/`: unit tests that do not require Blender.

## Requirements

- Python 3.10 or newer for the supervisor and tests.
- Blender 3.5.x with NumPy available to Blender Python.
- A dataset root containing `animation/inventory.json` and the source archives
  referenced by that inventory.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Run

Set installation-specific paths through environment variables rather than
editing the scripts:

```bash
export TEXVERSE_ROOT=/path/to/TexVerse-Skeleton-Animation
export BLENDER=/path/to/blender
# Optional, for portable Blender builds that need extra shared libraries:
export BLENDER_LIBRARY_DIRS=/path/to/libs:/path/to/mesa

./run_texverse_animation_batch.sh
```

The supervisor preserves an existing processed animation until the replacement
worker has completed successfully. Status reuse requires both the current
pipeline version and an exact match of output-affecting options.
