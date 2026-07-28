#!/usr/bin/env python3
"""Resumable supervisor for downloaded TexVerse FBX/GLB assets."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_logic import PIPELINE_VERSION


DEFAULT_ROOT = Path(os.environ.get("TEXVERSE_ROOT", SCRIPT_DIR.parent))
DEFAULT_BLENDER = os.environ.get("BLENDER", "blender")
DEFAULT_LIBRARY_DIRS = [
    value for value in os.environ.get("BLENDER_LIBRARY_DIRS", "").split(os.pathsep) if value
]


def paths_for(record: dict, root: Path) -> tuple[Path, Path, Path]:
    shard = Path(record["archive"]).parent
    raw_root = root / "animation" / "raw" / shard / record["id"]
    processed_root = root / "animation" / "processed" / shard / record["id"]
    status_path = root / "animation" / "status" / shard / f"{record['id']}.json"
    return raw_root, processed_root, status_path


def write_status(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n")
    temporary.replace(path)


def extract_raw_asset(archive: Path, raw_root: Path, source_path: str, root: Path) -> Path:
    """Extract a complete archive atomically so restarts never consume partial data."""
    source = raw_root / source_path
    if source.is_file():
        return source
    if raw_root.exists():
        shutil.rmtree(raw_root)

    relative = raw_root.relative_to(root / "animation" / "raw")
    staging = root / "animation" / "raw" / ".staging" / relative
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(staging)

    staged_source = staging / source_path
    if not staged_source.is_file():
        raise RuntimeError(f"Archive did not produce expected source file: {source_path}")
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(raw_root)
    return source


def processing_config(args: argparse.Namespace) -> dict:
    return {
        "resolution": args.resolution,
        "clip_frames": args.clip_frames,
        "max_clips": args.max_clips,
        "max_fps": args.max_fps,
        "min_motion": args.min_motion,
        "min_image_change": args.min_image_change,
        "image_change_pixel_threshold": args.image_change_pixel_threshold,
        "render_threads": args.render_threads,
    }


def read_worker_result(log_path: Path) -> dict:
    marker = None
    with log_path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("TEXVERSE_RESULT="):
                marker = line
    if marker is None:
        raise RuntimeError(f"Worker exited without result marker; see {log_path}")
    return json.loads(marker.split("=", 1)[1])


def recover_or_remove_backup(processed_root: Path) -> None:
    backup = processed_root.with_name(f".{processed_root.name}.previous")
    if not backup.exists():
        return
    if processed_root.exists():
        shutil.rmtree(backup)
    else:
        backup.replace(processed_root)


def publish_processed_root(staged_root: Path, processed_root: Path) -> None:
    """Replace one animation only after its new output has completed."""
    processed_root.parent.mkdir(parents=True, exist_ok=True)
    backup = processed_root.with_name(f".{processed_root.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if processed_root.exists():
        processed_root.replace(backup)
    try:
        staged_root.replace(processed_root)
    except Exception:
        if backup.exists() and not processed_root.exists():
            backup.replace(processed_root)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def base_status(record: dict, args: argparse.Namespace, gpu_id: str | None) -> dict:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "processing_config": processing_config(args),
        "id": record["id"],
        "archive": record["archive"],
        "gpu_id": gpu_id,
    }


def process_record(record: dict, args: argparse.Namespace, gpu_id: str | None) -> dict:
    root = args.root
    raw_root, processed_root, status_path = paths_for(record, root)
    shard = Path(record["archive"]).parent
    staging_output = root / "animation" / ".processed_staging" / record["id"]
    staged_processed_root = staging_output / shard / record["id"]
    log_path = root / "animation" / "logs" / shard / f"{record['id']}.log"

    try:
        recover_or_remove_backup(processed_root)
        archive = root / record["archive"]
        source = extract_raw_asset(archive, raw_root, record["source_path"], root)
        if staging_output.exists():
            shutil.rmtree(staging_output)
        staging_output.mkdir(parents=True, exist_ok=True)

        environment = dict(os.environ)
        if args.library_dir:
            current = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(args.library_dir + [current])
        if gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id

        command = [
            args.blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(args.worker),
            "--",
            "--source",
            str(source),
            "--sample-id",
            record["id"],
            "--source-id",
            record["id"],
            "--source-archive",
            record["archive"],
            "--output-root",
            str(staging_output),
            "--raw-root",
            str(root / "animation" / "raw"),
            "--resolution",
            str(args.resolution),
            "--clip-frames",
            str(args.clip_frames),
            "--max-clips",
            str(args.max_clips),
            "--max-fps",
            str(args.max_fps),
            "--min-motion",
            str(args.min_motion),
            "--min-image-change",
            str(args.min_image_change),
            "--image-change-pixel-threshold",
            str(args.image_change_pixel_threshold),
            "--render-threads",
            str(args.render_threads),
        ]

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_handle:
            result = subprocess.run(
                command,
                env=environment,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"Blender exited {result.returncode}; see {log_path}")

        worker_result = read_worker_result(log_path)
        if worker_result["status"] == "skipped_no_valid_clips":
            if processed_root.exists():
                shutil.rmtree(processed_root)
            write_status(status_path, {
                **base_status(record, args, gpu_id),
                "status": "skipped_no_valid_clips",
                "clips": worker_result["clips"],
                "raw_root": str(raw_root),
            })
            return {"status": "skipped_no_valid_clips", "id": record["id"], "gpu_id": gpu_id}

        if worker_result["status"] == "failed_no_valid_clips":
            errors = [
                {"clip_index": clip["clip_index"], "error": clip.get("error")}
                for clip in worker_result["clips"]
                if clip["status"] == "failed"
            ]
            raise RuntimeError(f"No valid clips exported: {errors}")
        if worker_result["status"] != "complete":
            raise RuntimeError(f"Unknown worker result: {worker_result['status']}")

        manifests = sorted(staged_processed_root.glob("*/source_manifest.json"))
        if len(manifests) != worker_result["clip_count"]:
            raise RuntimeError(
                f"Worker reported {worker_result['clip_count']} clips but wrote {len(manifests)}"
            )
        publish_processed_root(staged_processed_root, processed_root)
        write_status(status_path, {
            **base_status(record, args, gpu_id),
            "status": "complete",
            "processed_root": str(processed_root),
            "raw_root": str(raw_root),
            "clip_count": len(manifests),
            "candidate_clip_count": worker_result["candidate_clip_count"],
            "clip_statuses": [clip["status"] for clip in worker_result["clips"]],
        })
        return {
            "status": "complete",
            "id": record["id"],
            "gpu_id": gpu_id,
            "clip_count": len(manifests),
        }
    except Exception as error:
        write_status(status_path, {
            **base_status(record, args, gpu_id),
            "status": "failed",
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        return {"status": "failed", "id": record["id"], "gpu_id": gpu_id, "error": str(error)}
    finally:
        if staging_output.exists():
            shutil.rmtree(staging_output)


def prior_status_is_reusable(
    prior: dict,
    args: argparse.Namespace,
    processed_root: Path,
) -> bool:
    if prior.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if prior.get("processing_config") != processing_config(args):
        return False
    if prior.get("status") == "complete":
        manifests = sorted(processed_root.glob("*/source_manifest.json"))
        return len(manifests) == prior.get("clip_count") and bool(manifests)
    if prior.get("status") == "failed":
        return not args.retry_failed
    return prior.get("status") == "skipped_no_valid_clips"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--worker", type=Path, default=SCRIPT_DIR / "export_texverse_animation.py")
    parser.add_argument("--library-dir", action="append", default=DEFAULT_LIBRARY_DIRS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--max-fps", type=float, default=16.0)
    parser.add_argument("--min-motion", type=float, default=0.01)
    parser.add_argument("--min-image-change", type=float, default=0.001)
    parser.add_argument("--image-change-pixel-threshold", type=int, default=10)
    parser.add_argument("--render-threads", type=int, default=12)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-ids", default="")
    args = parser.parse_args()
    if args.inventory is None:
        args.inventory = args.root / "animation" / "inventory.json"
    return args


def main() -> None:
    args = parse_args()
    records = json.loads(args.inventory.read_text())["processable_records"]
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]

    candidates = []
    skipped = 0
    for record in records:
        _, processed_root, status_path = paths_for(record, args.root)
        if status_path.is_file() and not args.force:
            prior = json.loads(status_path.read_text())
            if prior_status_is_reusable(prior, args, processed_root):
                skipped += 1
                continue
        candidates.append(record)
        if args.limit and len(candidates) >= args.limit:
            break

    completed = failed = skipped_no_valid_clips = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_record,
                record,
                args,
                gpu_ids[index % len(gpu_ids)] if gpu_ids else None,
            )
            for index, record in enumerate(candidates)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result["status"] == "complete":
                completed += 1
            elif result["status"] == "skipped_no_valid_clips":
                skipped_no_valid_clips += 1
            else:
                failed += 1
            print(json.dumps({"progress": {
                "completed": completed,
                "skipped_no_valid_clips": skipped_no_valid_clips,
                "failed": failed,
                "queued": len(candidates),
                "id": result["id"],
                "gpu_id": result.get("gpu_id"),
            }}), flush=True)
    print(json.dumps({
        "completed": completed,
        "skipped_no_valid_clips": skipped_no_valid_clips,
        "failed": failed,
        "skipped": skipped,
    }))


if __name__ == "__main__":
    main()
