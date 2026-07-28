#!/usr/bin/env sh
# Resume the TexVerse export after a Pod restart; completed current-version animations are skipped.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=${TEXVERSE_ROOT:-$(dirname -- "$SCRIPT_DIR")}
cd "$ROOT"
exec python "$SCRIPT_DIR/process_texverse_animation.py" \
  --root "$ROOT" \
  --blender "${BLENDER:-blender}" \
  --workers 4 \
  --resolution 512 \
  --clip-frames 16 \
  --max-clips 6 \
  --max-fps 16 \
  --min-motion 0.01 \
  --min-image-change 0.001 \
  --image-change-pixel-threshold 10 \
  --render-threads 12
