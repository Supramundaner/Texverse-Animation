#!/usr/bin/env python3
"""Export one normalized TexVerse FBX/GLB asset into LTA-style mesh frames.

Run with Blender. The asset is normalized as a whole. Each exported reference
and the first target frame of each clip are centered at the origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_logic import (
    CANONICAL_AXIS_ROTATIONS,
    PIPELINE_VERSION,
    reference_bbox_normalization_scale,
    sampled_orbit_camera_indices,
    sampled_lighting_preset_index,
    split_clips,
)
from geometry_logic import best_axis_aligned_rotation, origin_centering_translation


NORMALIZED_MAX_EXTENT = 2.0
ORIGIN_CENTROID_TOLERANCE = 1e-5


def matrix_to_list(matrix: Matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def transform_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to an exported vertex array."""
    values = np.asarray(vertices, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    homogeneous = np.concatenate(
        (values, np.ones((len(values), 1), dtype=np.float64)), axis=1
    )
    return (homogeneous @ matrix.T)[:, :3].astype(np.float32)


def transform_skeleton_matrices(matrices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply an output-space world transform to global joint matrices."""
    return np.matmul(transform[np.newaxis, :, :], matrices).astype(np.float32)


def centroid64(vertices: np.ndarray) -> np.ndarray:
    return np.asarray(vertices).mean(axis=0, dtype=np.float64)


def require_origin_centroid(label: str, vertices: np.ndarray) -> np.ndarray:
    centroid = centroid64(vertices)
    displacement = float(np.linalg.norm(centroid))
    if displacement > ORIGIN_CENTROID_TOLERANCE:
        raise RuntimeError(
            f"{label} centroid is not at the origin: {centroid.tolist()} "
            f"(norm={displacement}, tolerance={ORIGIN_CENTROID_TOLERANCE})"
        )
    return centroid


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--published-output-root", type=Path)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--max-fps", type=float, default=16.0)
    parser.add_argument("--min-motion", type=float, default=0.01)
    parser.add_argument("--max-centroid-motion-bbox-ratio", type=float, default=1.0)
    parser.add_argument("--min-image-change", type=float, default=0.01)
    parser.add_argument("--image-change-pixel-threshold", type=int, default=10)
    parser.add_argument("--min-reference-foreground", type=float, default=0.05)
    parser.add_argument("--reference-background-pixel-threshold", type=int, default=10)
    parser.add_argument("--render-threads", type=int, default=12)
    return parser.parse_args(argv)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def imported_scene(source: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if source.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(source), use_anim=True)
    elif source.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(source))
    else:
        raise RuntimeError(f"Unsupported source format: {source}")


SCENE_MESH_NAME = re.compile(r"(?:^|[_. -])(ground|floor|plane\d*|terrain|stage|backdrop|background|environment|scene)(?:$|[_. -])")


def has_armature_binding(obj, armatures: set) -> bool:
    current = obj.parent
    while current:
        if current in armatures:
            return True
        current = current.parent
    return any(modifier.type == "ARMATURE" and modifier.object in armatures for modifier in obj.modifiers)


def mesh_bounds(obj) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray([obj.matrix_world @ Vector(corner) for corner in obj.bound_box], dtype=np.float32)
    return corners.min(axis=0), corners.max(axis=0)


def is_scene_mesh(obj, mesh_scale_median: float) -> bool:
    if SCENE_MESH_NAME.search(obj.name.lower()):
        return True
    minimum, maximum = mesh_bounds(obj)
    extent = maximum - minimum
    largest = float(extent.max())
    smallest = float(extent.min())
    # A very large, nearly flat standalone plane is set dressing, not character geometry.
    return largest > 4.0 * max(mesh_scale_median, 1e-6) and smallest < 0.01 * largest


def select_meshes(scene) -> tuple[list, dict]:
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and not obj.hide_render]
    if not meshes:
        raise RuntimeError("No renderable mesh objects were imported")
    armatures = {obj for obj in scene.objects if obj.type == "ARMATURE"}
    armature_meshes = [obj for obj in meshes if has_armature_binding(obj, armatures)]
    if armature_meshes:
        selected = armature_meshes
        method = "armature_associated_meshes"
    else:
        scales = []
        for obj in meshes:
            minimum, maximum = mesh_bounds(obj)
            scales.append(float((maximum - minimum).max()))
        median_scale = float(np.median(scales))
        selected = [obj for obj in meshes if not is_scene_mesh(obj, median_scale)]
        method = "all_meshes_excluding_named_or_large_thin_scene_meshes"
    if not selected:
        selected = meshes
        method = "all_renderable_meshes_fallback_scene_filter_would_remove_everything"
    removed = sorted(obj.name for obj in meshes if obj not in selected)
    for obj in meshes:
        if obj not in selected:
            bpy.data.objects.remove(obj, do_unlink=True)
    return sorted(selected, key=lambda obj: obj.name), {
        "method": method,
        "input_mesh_count": len(meshes),
        "selected_mesh_count": len(selected),
        "removed_mesh_names": removed,
    }


def force_opaque_materials(meshes: list) -> int:
    """Avoid invisible FBX materials whose Principled alpha imports as zero."""
    changed = 0
    materials = {material for obj in meshes for material in obj.data.materials if material}
    for material in materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            alpha = node.inputs.get("Alpha")
            if alpha is None:
                continue
            for link in list(alpha.links):
                material.node_tree.links.remove(link)
            if alpha.default_value != 1.0:
                alpha.default_value = 1.0
                changed += 1
    return changed


def set_armature_pose_position(scene, position: str) -> list[str]:
    """Set every imported armature to REST or POSE and report the affected rigs."""
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    for armature in armatures:
        armature.data.pose_position = position
    # Re-evaluate active actions after changing the Armature data mode.
    scene.frame_set(scene.frame_current)
    bpy.context.view_layer.update()
    return sorted(armature.name for armature in armatures)


def skeleton_descriptor(scene, meshes: list) -> dict | None:
    """Describe one deterministic mesh-bound armature for per-frame export."""
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    bound_armatures = [
        armature for armature in armatures
        if any(has_armature_binding(mesh, {armature}) for mesh in meshes)
    ]
    candidates = sorted(bound_armatures, key=lambda obj: obj.name)
    if not candidates:
        return None
    armature = candidates[0]
    bones = sorted(armature.data.bones, key=lambda bone: bone.name)
    if not bones:
        return None
    bone_indices = {bone.name: index for index, bone in enumerate(bones)}
    return {
        "armature": armature.name,
        "space": "normalized_blender_world",
        "format": "per_frame_global_joint_matrices",
        "rest_matrix_space": "armature_local",
        "joint_count": len(bones),
        "joints": [
            {
                "name": bone.name,
                "parent_index": bone_indices.get(bone.parent.name) if bone.parent else -1,
                "rest_matrix_local": matrix_to_list(bone.matrix_local),
            }
            for bone in bones
        ],
    }


def evaluated_skeleton_matrices(scene, descriptor: dict) -> np.ndarray:
    """Read global normalized-world pose matrices in descriptor joint order."""
    armature = scene.objects[descriptor["armature"]]
    matrices = []
    for joint in descriptor["joints"]:
        pose_bone = armature.pose.bones[joint["name"]]
        matrices.append(np.asarray(armature.matrix_world @ pose_bone.matrix, dtype=np.float32))
    values = np.asarray(matrices, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite skeleton transform in armature {armature.name}")
    return values


def root_joint_alignment(scene, meshes: list) -> tuple[Matrix, dict]:
    """Select the shallowest animated bone for reference-orientation alignment."""
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    bound_armatures = [
        armature for armature in armatures
        if any(has_armature_binding(mesh, {armature}) for mesh in meshes)
    ]
    candidates = sorted(bound_armatures or armatures, key=lambda obj: obj.name)
    if not candidates:
        return Matrix.Identity(4), {"applied": False, "reason": "no_armature"}
    armature = candidates[0]
    roots = sorted((bone for bone in armature.data.bones if bone.parent is None), key=lambda bone: bone.name)
    if not roots:
        return Matrix.Identity(4), {"applied": False, "reason": "armature_has_no_root_bone", "armature": armature.name}

    animated_names = set()
    action = armature.animation_data.action if armature.animation_data else None
    if action:
        for curve in action.fcurves:
            match = re.match(r'^pose\.bones\["(.+)"\]\.', curve.data_path)
            if match:
                animated_names.add(match.group(1))
    animated_bones = [bone for bone in armature.data.bones if bone.name in animated_names]

    def hierarchy_depth(bone) -> int:
        depth = 0
        current = bone.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    def descendant_count(bone) -> int:
        return sum(1 for candidate in armature.data.bones if bone in candidate.parent_recursive)

    if animated_bones:
        root = min(
            animated_bones,
            key=lambda bone: (hierarchy_depth(bone), -descendant_count(bone), bone.name),
        )
        selection_method = "shallowest_animated_bone_then_most_descendants"
    else:
        root = roots[0]
        selection_method = "root_bone_fallback_no_animated_bones"
    # Called in REST mode first, then again in POSE mode by the caller.
    rest_matrix = armature.matrix_world @ root.matrix_local
    return rest_matrix, {
        "applied": True,
        "armature": armature.name,
        "bone": root.name,
        "selection_method": selection_method,
        "hierarchy_depth": hierarchy_depth(root),
        "animated_bone_count": len(animated_bones),
    }


def root_joint_pose_matrix(scene, armature_name: str, bone_name: str) -> Matrix:
    armature = scene.objects[armature_name]
    pose_bone = armature.pose.bones[bone_name]
    return armature.matrix_world @ pose_bone.matrix


def canonical_reference_transform(
    scene,
    meshes: list,
    reference_vertices: np.ndarray,
    first_target_vertices: np.ndarray,
    target_centroid: np.ndarray,
) -> tuple[Matrix, dict]:
    """Align REST geometry to the first pose with one of 24 axis rotations."""
    rest_matrix, root_metadata = root_joint_alignment(scene, meshes)
    reference_centroid = centroid64(reference_vertices)
    scale_factor = 1.0
    rest_rotation = first_target_rotation = None
    pose_matrix = None

    if root_metadata.get("applied") is not False:
        pose_matrix = root_joint_pose_matrix(scene, root_metadata["armature"], root_metadata["bone"])
        rest_scale = np.asarray(rest_matrix.to_scale(), dtype=np.float32)
        pose_scale = np.asarray(pose_matrix.to_scale(), dtype=np.float32)
        ratios = pose_scale / np.maximum(rest_scale, 1e-8)
        # Use the root's uniform scale, or its geometric-average equivalent for noisy FBX matrices.
        scale_factor = float(np.exp(np.log(np.maximum(ratios, 1e-8)).mean()))

        rest_rotation = rest_matrix.to_quaternion().to_matrix()
        first_target_rotation = pose_matrix.to_quaternion().to_matrix()

    rotation_array, canonical_rotation_index, geometry_alignment = best_axis_aligned_rotation(
        reference_vertices,
        first_target_vertices,
        [candidate for candidate, _ in CANONICAL_AXIS_ROTATIONS],
    )
    canonical_target_axes = CANONICAL_AXIS_ROTATIONS[canonical_rotation_index][1]
    rotation = Matrix(rotation_array).to_4x4()
    method = "align_rest_mesh_to_first_target_by_best_of_24_axis_rotations_then_match_centroid"

    reference_centered = np.asarray(reference_vertices, dtype=np.float64) - reference_centroid
    target_centered = (
        np.asarray(first_target_vertices, dtype=np.float64)
        - centroid64(first_target_vertices)
    )
    scaled_aligned = scale_factor * (reference_centered @ rotation_array.T)
    geometry_alignment["rms_error_after_root_scale"] = float(
        np.sqrt(np.mean(np.sum((scaled_aligned - target_centered) ** 2, axis=1)))
    )

    transform = (
        Matrix.Translation(Vector(target_centroid))
        @ Matrix.Scale(scale_factor, 4)
        @ rotation
        @ Matrix.Translation(-Vector(reference_centroid))
    )
    metadata = {
        "applied": True,
        "method": method,
        "root": root_metadata,
        "rest_root_rotation": matrix_to_list(rest_rotation) if rest_rotation is not None else None,
        "first_target_root_rotation": matrix_to_list(first_target_rotation) if first_target_rotation is not None else None,
        "canonical_target_rotation": matrix_to_list(rotation_array),
        "canonical_target_axes": list(canonical_target_axes) if canonical_target_axes is not None else None,
        "canonical_rotation_index": canonical_rotation_index,
        "applied_rotation": matrix_to_list(rotation),
        "orientation_selection": geometry_alignment,
        "reference_uniform_scale": scale_factor,
        "root_rest_scale": [float(value) for value in rest_matrix.to_scale()],
        "root_first_target_scale": (
            [float(value) for value in pose_matrix.to_scale()] if pose_matrix is not None else None
        ),
        "reference_centroid_before": reference_centroid.astype(float).tolist(),
        "first_target_centroid_before_output_centering": (
            centroid64(first_target_vertices).astype(float).tolist()
        ),
        "first_target_centroid": target_centroid.astype(float).tolist(),
        "reference_transform": [[float(value) for value in row] for row in transform],
    }
    return transform, metadata


def apply_reference_transform(scene, transform: Matrix) -> list[tuple[object, Matrix]]:
    """Temporarily transform imported scene roots to render the aligned REST reference."""
    roots = [obj for obj in scene.objects if obj.parent is None and obj.type not in {"CAMERA", "LIGHT"}]
    originals = [(obj, obj.matrix_world.copy()) for obj in roots]
    for obj, original in originals:
        obj.matrix_world = transform @ original
    bpy.context.view_layer.update()
    return originals


def replace_applied_transform(
    originals: list[tuple[object, Matrix]], transform: Matrix
) -> None:
    """Replace a temporary scene-wide transform without accumulating error."""
    for obj, original in originals:
        obj.matrix_world = transform @ original
    bpy.context.view_layer.update()


def restore_reference_transform(originals: list[tuple[object, Matrix]]) -> None:
    for obj, original in originals:
        obj.matrix_world = original
    bpy.context.view_layer.update()


def normalize_import(scene, meshes: list, minimum: np.ndarray, maximum: np.ndarray) -> dict:
    """Apply one scene-wide transform derived from the unposed reference geometry."""
    center = 0.5 * (minimum + maximum)
    scale = NORMALIZED_MAX_EXTENT / max(float(np.max(maximum - minimum)), 1e-6)
    normalizer = bpy.data.objects.new("TexVerse_Normalizer", None)
    scene.collection.objects.link(normalizer)
    normalizer.location = tuple(-center * scale)
    normalizer.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    roots = [obj for obj in scene.objects if obj is not normalizer and obj.parent is None]
    for obj in roots:
        original_world = obj.matrix_world.copy()
        obj.parent = normalizer
        # Preserve each root's local animated transform, while applying the normalizer in world space.
        obj.matrix_world = normalizer.matrix_world @ original_world
    bpy.context.view_layer.update()
    return {
        "method": "parent_root_objects_translate_then_uniform_scale_from_reference_mesh",
        "source_bounds_min": minimum.astype(float).tolist(),
        "source_bounds_max": maximum.astype(float).tolist(),
        "source_center": center.astype(float).tolist(),
        "uniform_scale": float(scale),
        "normalized_max_extent": NORMALIZED_MAX_EXTENT,
        "root_joint_alignment": "none",
    }


def evaluated_geometry(meshes: list, depsgraph) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    vertex_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    topology = []
    offset = 0
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        try:
            coordinates = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", coordinates)
            vertices = coordinates.reshape(-1, 3)
            homogeneous = np.concatenate((vertices, np.ones((len(vertices), 1), dtype=np.float32)), axis=1)
            matrix = np.asarray(evaluated.matrix_world, dtype=np.float32)
            world_vertices = (homogeneous @ matrix.T)[:, :3]
            if not np.isfinite(world_vertices).all():
                raise RuntimeError(f"Non-finite evaluated vertex coordinates in mesh {obj.name}")
            mesh.calc_loop_triangles()
            faces = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
            mesh.loop_triangles.foreach_get("vertices", faces)
            faces = faces.reshape(-1, 3) + offset
            vertex_parts.append(world_vertices.astype(np.float32, copy=False))
            face_parts.append(faces)
            topology.append({"name": obj.name, "vertex_offset": offset, "vertex_count": len(world_vertices), "face_count": len(faces)})
            offset += len(world_vertices)
        finally:
            evaluated.to_mesh_clear()
    return np.concatenate(vertex_parts), np.concatenate(face_parts), topology


def evaluated_motion_state(meshes: list, depsgraph, topology: list[dict], vertex_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read sampled vertices plus the exact all-vertex centroid for translation-invariant motion."""
    values = np.empty((len(vertex_indices), 3), dtype=np.float32)
    centroid_sum = np.zeros(3, dtype=np.float64)
    total_vertices = 0
    for object_index, (obj, expected) in enumerate(zip(meshes, topology)):
        start = expected["vertex_offset"]
        stop = start + expected["vertex_count"]
        requested = np.where((vertex_indices >= start) & (vertex_indices < stop))[0]
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        try:
            if len(mesh.vertices) != expected["vertex_count"]:
                raise RuntimeError(f"Topology changed at mesh {object_index}")
            coordinates = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", coordinates)
            local_all = coordinates.reshape(-1, 3)
            matrix = np.asarray(evaluated.matrix_world, dtype=np.float32)
            if not np.isfinite(local_all).all() or not np.isfinite(matrix).all():
                raise RuntimeError(f"Non-finite evaluated vertex coordinates in mesh {obj.name}")
            local_mean = local_all.mean(axis=0, dtype=np.float64)
            centroid_sum += (np.append(local_mean, 1.0) @ matrix.T)[:3] * len(local_all)
            total_vertices += len(local_all)
            if not len(requested):
                continue
            local_indices = vertex_indices[requested] - start
            local = local_all[local_indices]
            homogeneous = np.concatenate((local, np.ones((len(local), 1), dtype=np.float32)), axis=1)
            values[requested] = (homogeneous @ matrix.T)[:, :3]
        finally:
            evaluated.to_mesh_clear()
    centroid = (centroid_sum / max(total_vertices, 1)).astype(np.float32)
    if not np.isfinite(values).all() or not np.isfinite(centroid).all():
        raise RuntimeError("Non-finite evaluated sampled vertices or mesh centroid")
    return values, centroid


def measure_motion(
    meshes: list,
    depsgraph,
    topology: list[dict],
    frames: list[int],
    vertex_indices: np.ndarray,
    reference_bbox_min: np.ndarray,
    reference_bbox_max: np.ndarray,
) -> dict:
    reference_bbox_extent = reference_bbox_max - reference_bbox_min
    reference_bbox_scale = reference_bbox_normalization_scale(
        reference_bbox_min, reference_bbox_max
    )
    previous_vertices = previous_centroid = None
    first_centroid = None
    relative_steps: list[float] = []
    global_centroid_steps: list[float] = []
    global_centroid_from_first: list[float] = []
    for source_frame in frames:
        bpy.context.scene.frame_set(source_frame)
        bpy.context.view_layer.update()
        current_vertices, current_centroid = evaluated_motion_state(meshes, depsgraph, topology, vertex_indices)
        if first_centroid is None:
            first_centroid = current_centroid.copy()
        global_centroid_from_first.append(float(np.linalg.norm(current_centroid - first_centroid)))
        if previous_vertices is not None:
            # Remove whole-mesh translation before assessing deformation / articulated motion.
            previous_relative = previous_vertices - previous_centroid
            current_relative = current_vertices - current_centroid
            relative_steps.append(float(
                np.linalg.norm(
                    (current_relative - previous_relative) / reference_bbox_scale,
                    axis=1,
                ).mean()
            ))
            global_centroid_steps.append(float(np.linalg.norm(current_centroid - previous_centroid)))
        previous_vertices, previous_centroid = current_vertices, current_centroid
    return {
        "method": "mean_euclidean_displacement_normalized_by_reference_mesh_bbox_largest_edge_after_subtracting_each_frame_full_mesh_centroid",
        "frame_definition": "consecutive_exported_frames_after_fps_downsampling_and_frame_window_selection",
        "vertex_sample_count": int(len(vertex_indices)),
        "vertex_indices": vertex_indices.astype(int).tolist(),
        "reference_bbox_min": reference_bbox_min.astype(float).tolist(),
        "reference_bbox_max": reference_bbox_max.astype(float).tolist(),
        "reference_bbox_extent": reference_bbox_extent.astype(float).tolist(),
        "reference_bbox_normalization_scale": float(reference_bbox_scale),
        "step_count": len(relative_steps),
        "mean_per_point_per_frame": float(np.mean(relative_steps)) if relative_steps else 0.0,
        "step_mean_displacements": relative_steps,
        "global_centroid_motion_mean_per_frame": float(np.mean(global_centroid_steps)) if global_centroid_steps else 0.0,
        "global_centroid_step_displacements": global_centroid_steps,
        "global_centroid_max_displacement_from_first": max(global_centroid_from_first, default=0.0),
        "space": "normalized_blender_world",
    }


def effective_animation_range(scene, meshes: list) -> tuple[int, int, dict]:
    """Use active animation actions rather than scene padding/hold frames."""
    actions = []
    candidates = list(meshes) + [obj for obj in scene.objects if obj.type == "ARMATURE"]
    for obj in candidates:
        if obj.animation_data and obj.animation_data.action:
            actions.append((f"object:{obj.name}", obj.animation_data.action))
        if obj.type == "MESH" and obj.data.shape_keys:
            animation_data = obj.data.shape_keys.animation_data
            if animation_data and animation_data.action:
                actions.append((f"shape_keys:{obj.name}", animation_data.action))
    if not actions:
        return scene.frame_start, scene.frame_end, {
            "method": "scene_frame_range_no_active_actions",
            "scene_frame_range": [scene.frame_start, scene.frame_end],
            "action_ranges": [],
        }
    starts = [int(math.floor(action.frame_range[0])) for _, action in actions]
    ends = [int(math.ceil(action.frame_range[1])) for _, action in actions]
    start, end = max(scene.frame_start, min(starts)), min(scene.frame_end, max(ends))
    return start, max(start, end), {
        "method": "union_of_active_action_keyframe_ranges",
        "scene_frame_range": [scene.frame_start, scene.frame_end],
        "action_ranges": [
            {"owner": owner, "action": action.name, "frame_range": list(action.frame_range)}
            for owner, action in actions
        ],
    }


def sampled_frames(scene, maximum_fps: float, frame_start: int, frame_end: int) -> tuple[list[int], float, dict]:
    source_fps = float(scene.render.fps) if scene.render.fps else 24.0
    stride = max(1, math.ceil(source_fps / maximum_fps))
    candidates = list(range(frame_start, frame_end + 1, stride))
    if not candidates:
        candidates = [frame_start]
    return candidates, source_fps / stride, {
        "method": "all_effective_frames_after_fps_downsampling",
        "source_fps": source_fps,
        "stride": stride,
        "max_fps": maximum_fps,
        "effective_frame_range": [frame_start, frame_end],
    }


CAMERA_COUNT = 12
LIGHTING_PRESETS = (
    {
        "name": "balanced_ring",
        "lights": (
            ("front", (0.0, -1.0, 0.65), 360.0),
            ("back", (0.0, 1.0, 0.65), 300.0),
            ("left", (-1.0, 0.0, 0.65), 330.0),
            ("right", (1.0, 0.0, 0.65), 330.0),
            ("top", (0.0, 0.0, 1.0), 400.0),
        ),
    },
    {
        "name": "front_key",
        "lights": (
            ("front", (0.0, -1.0, 0.8), 700.0),
            ("back", (0.0, 1.0, 0.55), 140.0),
            ("left", (-1.0, 0.0, 0.65), 280.0),
            ("right", (1.0, 0.0, 0.65), 180.0),
            ("top", (0.0, 0.0, 1.0), 420.0),
        ),
    },
    {
        "name": "side_key",
        "lights": (
            ("front", (0.0, -1.0, 0.65), 260.0),
            ("back", (0.0, 1.0, 0.55), 180.0),
            ("left", (-1.0, 0.0, 0.7), 700.0),
            ("right", (1.0, 0.0, 0.65), 240.0),
            ("top", (0.0, 0.0, 1.0), 400.0),
        ),
    },
    {
        "name": "top_key",
        "lights": (
            ("front", (0.0, -1.0, 0.65), 320.0),
            ("back", (0.0, 1.0, 0.65), 260.0),
            ("left", (-1.0, 0.0, 0.65), 260.0),
            ("right", (1.0, 0.0, 0.65), 260.0),
            ("top", (0.0, 0.0, 1.0), 850.0),
        ),
    },
)


def sampled_camera_indices(sample_id: str, clip_index: int) -> list[int]:
    """Always include front camera_0 and deterministically sample one side view."""
    return sampled_orbit_camera_indices(sample_id, clip_index, CAMERA_COUNT)


def camera_view(camera_index: int) -> dict:
    """Return one of twelve evenly spaced views around the world Z axis."""
    if not 0 <= camera_index < CAMERA_COUNT:
        raise ValueError(f"camera index must be between 0 and {CAMERA_COUNT - 1}")
    angle_degrees = camera_index * 30.0
    angle = math.radians(angle_degrees)
    offset = np.asarray((math.sin(angle), -math.cos(angle), 0.0), dtype=np.float32)
    right = np.asarray((math.cos(angle), math.sin(angle), 0.0), dtype=np.float32)
    up = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    return {
        "index": camera_index,
        "name": f"camera_{camera_index}",
        "angle_degrees": angle_degrees,
        "offset": offset,
        "right": right,
        "up": up,
        "depth": offset,
    }


def effective_camera_half_tangents(scene, camera_data) -> tuple[float, float]:
    """Return the rendered frame's horizontal and vertical half-FOV tangents."""
    frame = camera_data.view_frame(scene=scene)
    horizontal = max(abs(corner.x / corner.z) for corner in frame if abs(corner.z) > 1e-8)
    vertical = max(abs(corner.y / corner.z) for corner in frame if abs(corner.z) > 1e-8)
    return horizontal, vertical


def camera_distance_for_bounds(
    scene,
    camera_data,
    minimum: np.ndarray,
    maximum: np.ndarray,
    view: dict,
) -> float:
    """Fit a perspective camera to projected bounds with five percent empty space per edge."""
    half_extent = 0.5 * (maximum - minimum)
    usable_half_image = 0.90  # Five percent empty space on each image edge.
    horizontal_tangent, vertical_tangent = effective_camera_half_tangents(scene, camera_data)
    horizontal_half_extent = float(np.dot(np.abs(view["right"]), half_extent))
    vertical_half_extent = float(np.dot(np.abs(view["up"]), half_extent))
    depth_half_extent = float(np.dot(np.abs(view["depth"]), half_extent))
    horizontal = horizontal_half_extent / max(horizontal_tangent * usable_half_image, 1e-6)
    vertical = vertical_half_extent / max(vertical_tangent * usable_half_image, 1e-6)
    # The nearest depth bound, rather than the mesh centre, controls perspective cropping.
    return float(depth_half_extent + max(horizontal, vertical, 0.001))


def configure_render(
    scene, resolution: int, vertices: np.ndarray, render_threads: int, lighting_preset: dict
):
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    camera_data = bpy.data.cameras.new("TexVerse_Camera")
    camera = bpy.data.objects.new("TexVerse_Camera", camera_data)
    scene.collection.objects.link(camera)
    camera["texverse_export_helper"] = True
    camera_data.type = "PERSP"
    camera_data.lens = 50.0
    view = camera_view(0)
    distance = camera_distance_for_bounds(scene, camera_data, minimum, maximum, view)
    camera.location = tuple(center + view["offset"] * distance)
    camera.rotation_euler = (Vector(center) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.clip_start = 0.001
    camera_data.clip_end = 100.0
    scene.camera = camera
    lights = []
    for role, _, energy in lighting_preset["lights"]:
        name = f"TexVerse_{role.title()}"
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = 4.0
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light["texverse_export_helper"] = True
        light.location = tuple(center)
        light.rotation_euler = (Vector(center) - light.location).to_track_quat("-Z", "Y").to_euler()
        lights.append((light, role))
    scene.render.engine = "CYCLES"
    # CUDA and OptiX kernel initialization is not compatible with this Blender build on A800.
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.render.threads_mode = "FIXED"
    scene.render.threads = render_threads
    scene.render.image_settings.file_format = "PNG"
    scene.use_nodes = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("TexVerse_World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    return camera, lights


def remove_export_render_helpers(scene) -> None:
    """Drop a prior clip's camera/lights so clip illumination never accumulates."""
    for obj in list(scene.objects):
        if obj.get("texverse_export_helper"):
            bpy.data.objects.remove(obj, do_unlink=True)


def update_camera(
    scene, camera, lights: list, vertices: np.ndarray, view: dict, lighting_preset: dict
) -> dict:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    distance = camera_distance_for_bounds(scene, camera.data, minimum, maximum, view)
    camera.location = tuple(center + view["offset"] * distance)
    camera.rotation_euler = (Vector(center) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.clip_end = max(100.0, distance * 4.0)
    # Use the same world-space soft-light rig for every camera so view changes
    # do not also change the illumination distribution.
    extent = maximum - minimum
    light_radius = max(3.0, float(extent.max()) * 2.0)
    light_size = max(4.0, float(extent.max()) * 2.5)
    offsets = {
        role: np.asarray(direction, dtype=np.float32) * light_radius
        for role, direction, _ in lighting_preset["lights"]
    }
    for light, role in lights:
        light.data.size = light_size
        light.location = tuple(center + offsets[role])
        light.rotation_euler = (Vector(center) - light.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    return {
        "centroid": vertices.mean(axis=0).astype(float).tolist(),
        "bounds_min": minimum.astype(float).tolist(),
        "bounds_max": maximum.astype(float).tolist(),
        "camera_location": list(camera.location),
        "camera_rotation_euler": list(camera.rotation_euler),
        "camera_lens_mm": float(camera.data.lens),
        "view_direction": view["name"],
        "camera_index": int(view["index"]),
        "camera_name": view["name"],
        "camera_angle_degrees": float(view["angle_degrees"]),
    }


def bounds_metadata(vertices: np.ndarray) -> dict:
    return {
        "centroid": vertices.mean(axis=0).astype(float).tolist(),
        "bounds_min": vertices.min(axis=0).astype(float).tolist(),
        "bounds_max": vertices.max(axis=0).astype(float).tolist(),
    }


def build_reference_camera(aligned_reference: np.ndarray, target_camera: dict) -> dict:
    return {
        **target_camera,
        "camera_mode": "reference_from_reference_mesh_bbox",
        "reference_subject": bounds_metadata(aligned_reference),
    }


def build_frame_camera(vertices: np.ndarray, target_camera: dict) -> dict:
    return {
        **bounds_metadata(vertices),
        "camera_location": target_camera["camera_location"],
        "camera_rotation_euler": target_camera["camera_rotation_euler"],
        "camera_lens_mm": target_camera["camera_lens_mm"],
        "view_direction": target_camera["view_direction"],
        "camera_index": target_camera["camera_index"],
        "camera_name": target_camera["camera_name"],
        "camera_angle_degrees": target_camera["camera_angle_degrees"],
        "camera_mode": "fixed_from_clip_animation_union_bounds",
    }


def load_rendered_rgb8(path: Path) -> np.ndarray:
    """Load an output PNG through Blender as its encoded 8-bit RGB values."""
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        rgba = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(rgba)
        rgb = rgba.reshape((height, width, 4))[..., :3]
        return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    finally:
        bpy.data.images.remove(image)


def measure_image_change(image_paths: list[Path], pixel_threshold: int) -> dict:
    """Measure visible change as changed-pixel fractions between rendered frames."""
    changed_fractions = []
    mean_absolute_rgb_differences = []
    previous = None
    for path in image_paths:
        current = load_rendered_rgb8(path)
        if previous is not None:
            difference = np.abs(current.astype(np.int16) - previous.astype(np.int16))
            changed_fractions.append(float(np.mean(difference.max(axis=2) > pixel_threshold)))
            mean_absolute_rgb_differences.append(float(difference.mean()))
        previous = current
    return {
        "method": "mean_fraction_of_pixels_whose_max_rgb_channel_delta_exceeds_threshold",
        "comparison": "consecutive_rendered_target_frames",
        "pixel_space": "encoded_8bit_rgb_output_png",
        "pixel_difference_threshold": pixel_threshold,
        "step_count": len(changed_fractions),
        "mean_changed_pixel_fraction_per_frame": (
            float(np.mean(changed_fractions)) if changed_fractions else 0.0
        ),
        "max_changed_pixel_fraction": max(changed_fractions, default=0.0),
        "step_changed_pixel_fractions": changed_fractions,
        "mean_absolute_rgb_difference_per_frame": (
            float(np.mean(mean_absolute_rgb_differences)) if mean_absolute_rgb_differences else 0.0
        ),
        "step_mean_absolute_rgb_differences": mean_absolute_rgb_differences,
    }


def measure_reference_foreground(image_path: Path, pixel_threshold: int) -> dict:
    """Estimate reference foreground from its difference to the border background."""
    rgb = load_rendered_rgb8(image_path).astype(np.int16)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    difference = np.abs(rgb - background)
    foreground_fraction = float(np.mean(difference.max(axis=2) > pixel_threshold))
    return {
        "method": "fraction_of_pixels_different_from_median_border_background",
        "pixel_space": "encoded_8bit_rgb_output_png",
        "pixel_difference_threshold": pixel_threshold,
        "estimated_background_rgb": background.astype(float).tolist(),
        "foreground_pixel_fraction": foreground_fraction,
    }


def build_sample_row(
    args,
    clip_index: int,
    frame_index: int,
    source_frame: int,
    final_root: Path,
    reference_centroid: list[float],
    frame_centroid: list[float],
    motion: dict,
    vertex_file: str,
    image_file: str,
    camera_index: int,
    camera_name: str,
    skeleton_file: str | None,
) -> dict:
    return {
        "sample_id": args.sample_id,
        "clip_index": clip_index,
        "split": "train",
        "frame_index": frame_index,
        "source_frame": source_frame,
        "view_label": camera_name,
        "camera_index": camera_index,
        "camera_name": camera_name,
        "sample_root": str(final_root),
        "source_manifest": str(final_root / "source_manifest.json"),
        "rest_mesh": str(final_root / "reference_mesh.npz"),
        "target_vertices": str(final_root / "target_vertices" / vertex_file),
        "target_image": str(final_root / "target_images" / camera_name / image_file),
        "skeleton_motion": (
            str(final_root / "skeleton" / skeleton_file) if skeleton_file else None
        ),
        "reference_centroid": reference_centroid,
        "frame_centroid": frame_centroid,
        "clip_mean_motion_per_point_per_frame": motion["mean_per_point_per_frame"],
    }


def build_common_metadata(
    args,
    fps: float,
    sampling: dict,
    effective_range: dict,
    clip_selection: dict,
    raw_destination: Path,
    forced_opaque_materials: int,
    mesh_filter: dict,
    topology: list[dict],
    armatures: list[str],
    reference_source_frame: int,
    normalization: dict,
    skeleton: dict | None,
) -> dict:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "sample_id": args.sample_id,
        "source_id": args.source_id,
        "source_archive": args.source_archive,
        "source_file": str(args.source),
        "raw_asset_root": str(raw_destination),
        "format": args.source.suffix.lower()[1:],
        "fps": fps,
        "sampling": {
            **sampling,
            "effective_animation": effective_range,
            "clip_selection": clip_selection,
        },
        "render": {
            "resolution": args.resolution,
            "image_format": "png",
            "engine": "cycles",
            "device": "cpu",
            "render_threads": args.render_threads,
            "samples": 8,
            "camera": "perspective",
            "reference_camera_mode": "reference_from_reference_mesh_bbox",
            "target_camera_mode": "fixed_from_clip_animation_union_bounds",
            "target_camera_count": CAMERA_COUNT,
            "target_camera_sampling": "camera_0_plus_one_deterministic_view_from_camera_1_to_camera_11",
            "lighting": {
                "mode": "camera_independent_world_space_soft_lights",
                "preset_selection": "deterministic_sample_id_and_clip_index",
                "available_presets": [preset["name"] for preset in LIGHTING_PRESETS],
                "light_radius": "max(3.0, animation_bbox_largest_edge * 2.0)",
                "light_size": "max(4.0, animation_bbox_largest_edge * 2.5)",
                "world_color_linear_rgba": [0.0, 0.0, 0.0, 1.0],
                "world_strength": 0.0,
            },
            "forced_opaque_material_count": forced_opaque_materials,
        },
        "mesh_filter": mesh_filter,
        "objects": topology,
        "reference_pose": {
            "kind": "armature_rest_pose" if armatures else "static_import_pose_without_armature",
            "armatures": armatures,
            "source_frame": reference_source_frame,
        },
        "normalization": normalization,
        "skeleton": skeleton,
    }


def build_clip_manifest(
    common: dict,
    clip_index: int,
    frames: list[int],
    motion: dict,
    min_motion: float,
    image_change: dict,
    min_image_change: float,
    reference_foreground: dict,
    min_reference_foreground: float,
    aligned_reference: np.ndarray,
    faces: np.ndarray,
    final_root: Path,
    reference_alignment: dict,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    reference_camera: dict,
    target_camera: dict,
    target_cameras: dict[str, dict],
    sampled_camera_indices: list[int],
    frame_metadata: list[dict],
    frame_metadata_by_camera: dict[str, list[dict]],
    image_change_by_camera: dict[str, dict],
) -> dict:
    return {
        **common,
        "clip_index": clip_index,
        "exported_frame_count": len(frames),
        "exported_source_frames": frames,
        "motion": motion,
        "motion_filter": {"min_motion": min_motion, "passed": True},
        "centroid_motion_filter": {
            "max_centroid_motion_bbox_ratio": motion.get("max_centroid_motion_bbox_ratio"),
            "max_centroid_motion": motion.get("max_centroid_motion"),
            "measured_centroid_max_displacement_from_first": motion.get(
                "global_centroid_max_displacement_from_first"
            ),
            "passed": True,
        },
        "image_change": image_change,
        "image_change_filter": {
            "min_mean_changed_pixel_fraction_per_frame": min_image_change,
            "passed": True,
        },
        "reference_foreground": reference_foreground,
        "reference_foreground_filter": {
            "min_foreground_pixel_fraction": min_reference_foreground,
            "passed": True,
        },
        "vertex_count": int(len(aligned_reference)),
        "face_count": int(len(faces)),
        "reference_mesh": str(final_root / "reference_mesh.npz"),
        "reference_image": str(final_root / "reference_views" / "camera_0.png"),
        "reference_centroid": centroid64(aligned_reference).astype(float).tolist(),
        "reference_alignment": reference_alignment,
        "animation_union_bounds": {
            "min": bounds_min.astype(float).tolist(),
            "max": bounds_max.astype(float).tolist(),
        },
        "reference_camera": reference_camera,
        "target_camera": target_camera,
        "target_cameras": target_cameras,
        "sampled_camera_indices": sampled_camera_indices,
        "frames": frame_metadata,
        "frames_by_camera": frame_metadata_by_camera,
        "image_change_by_camera": image_change_by_camera,
        "skeleton_motion": (
            str(final_root / "skeleton") if common.get("skeleton") else None
        ),
    }


def export_clip(
    args, scene, meshes, depsgraph, faces, topology, reference_vertices,
    reference_source_frame, common, clip_index, frames,
):
    """Validate and export one independent clip under <animation-id>/<clip-index>."""
    lighting_preset = LIGHTING_PRESETS[
        sampled_lighting_preset_index(args.sample_id, clip_index, len(LIGHTING_PRESETS))
    ]
    clip_common = {
        **common,
        "render": {
            **common["render"],
            "lighting_preset": lighting_preset["name"],
            "lighting_area_lights": [
                {"role": role, "direction": list(direction), "energy": energy}
                for role, direction, energy in lighting_preset["lights"]
            ],
        },
    }
    remove_export_render_helpers(scene)
    set_armature_pose_position(scene, "POSE")
    animation_vertices = []
    animation_skeleton = []
    skeleton = common.get("skeleton")
    for source_frame in frames:
        scene.frame_set(source_frame)
        bpy.context.view_layer.update()
        vertices, frame_faces, current_topology = evaluated_geometry(meshes, depsgraph)
        if current_topology != topology or not np.array_equal(frame_faces, faces):
            raise RuntimeError(f"Topology changed in clip {clip_index} at frame {source_frame}")
        animation_vertices.append(vertices)
        if skeleton:
            animation_skeleton.append(evaluated_skeleton_matrices(scene, skeleton))
    first_vertices = animation_vertices[0]
    # Reference scale/alignment is clip-specific and must be known before
    # motion is normalized. Derive the bbox from the final exported reference,
    # rather than from the animation's union bounds.
    scene.frame_set(frames[0])
    bpy.context.view_layer.update()
    reference_transform, reference_alignment = canonical_reference_transform(
        scene, meshes, reference_vertices, first_vertices, np.zeros(3, dtype=np.float32)
    )
    reference_alignment["first_target_source_frame"] = frames[0]
    transform_array = np.asarray(reference_transform, dtype=np.float32)
    reference_homogeneous = np.concatenate(
        (reference_vertices, np.ones((len(reference_vertices), 1), dtype=np.float32)), axis=1
    )
    aligned_reference_for_motion = (reference_homogeneous @ transform_array.T)[:, :3]
    reference_bbox_min = aligned_reference_for_motion.min(axis=0)
    reference_bbox_max = aligned_reference_for_motion.max(axis=0)

    rng = np.random.default_rng(int(hashlib.sha256(f"{args.sample_id}:{clip_index}".encode()).hexdigest()[:16], 16))
    indices = np.sort(rng.choice(len(reference_vertices), size=min(100, len(reference_vertices)), replace=False))
    motion = measure_motion(
        meshes,
        depsgraph,
        topology,
        frames,
        indices,
        reference_bbox_min,
        reference_bbox_max,
    )
    if motion["mean_per_point_per_frame"] <= args.min_motion:
        return {"status": "skipped_low_motion", "clip_index": clip_index, "motion": motion, "frames": len(frames)}
    centroid_motion_limit = (
        motion["reference_bbox_normalization_scale"]
        * args.max_centroid_motion_bbox_ratio
    )
    centroid_motion_filter = {
        "max_centroid_motion_bbox_ratio": args.max_centroid_motion_bbox_ratio,
        "max_centroid_motion": centroid_motion_limit,
        "measured_centroid_max_displacement_from_first": motion[
            "global_centroid_max_displacement_from_first"
        ],
        "passed": motion["global_centroid_max_displacement_from_first"] <= centroid_motion_limit,
    }
    motion["max_centroid_motion_bbox_ratio"] = float(args.max_centroid_motion_bbox_ratio)
    motion["max_centroid_motion"] = float(centroid_motion_limit)
    if not centroid_motion_filter["passed"]:
        return {
            "status": "skipped_excessive_centroid_motion",
            "clip_index": clip_index,
            "frames": len(frames),
            "motion": motion,
            "centroid_motion_filter": centroid_motion_filter,
        }
    target_translation = np.eye(4, dtype=np.float64)
    target_translation[:3, 3] = origin_centering_translation(first_vertices)
    exported_animation_vertices = [
        transform_vertices(vertices, target_translation) for vertices in animation_vertices
    ]
    require_origin_centroid(f"clip {clip_index} target frame 0", exported_animation_vertices[0])
    exported_animation_skeleton = [
        transform_skeleton_matrices(matrices, target_translation)
        for matrices in animation_skeleton
    ]
    bounds_min = np.min([vertices.min(axis=0) for vertices in exported_animation_vertices], axis=0)
    bounds_max = np.max([vertices.max(axis=0) for vertices in exported_animation_vertices], axis=0)
    bounds_vertices = np.stack((bounds_min, bounds_max))
    target_camera, target_lights = configure_render(
        scene, args.resolution, bounds_vertices, args.render_threads, lighting_preset
    )
    sampled_indices = sampled_camera_indices(args.sample_id, clip_index)
    target_camera_metadata_by_name = {}
    for camera_index in sampled_indices:
        view = camera_view(camera_index)
        metadata = update_camera(
            scene, target_camera, target_lights, bounds_vertices, view, lighting_preset
        )
        target_camera_metadata_by_name[view["name"]] = metadata
    target_camera_metadata = target_camera_metadata_by_name["camera_0"]
    final_root = args.output_root / Path(args.source_archive).parent / args.sample_id / str(clip_index)
    published_output_root = args.published_output_root or args.output_root
    manifest_root = (
        published_output_root
        / Path(args.source_archive).parent
        / args.sample_id
        / str(clip_index)
    )
    staging = args.output_root / ".staging" / Path(args.source_archive).parent / args.sample_id / str(clip_index)
    if staging.exists():
        shutil.rmtree(staging)
    vertices_dir = staging / "target_vertices"
    images_dir = staging / "target_images"
    reference_dir = staging / "reference_views"
    skeleton_dir = staging / "skeleton"
    for directory in (vertices_dir, images_dir, reference_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if skeleton:
        skeleton_dir.mkdir(parents=True, exist_ok=True)
        write_json(skeleton_dir / "metadata.json", {
            **skeleton,
            "source_frames": frames,
            "frame_file_pattern": "frame_%04d.npy",
            "dtype": "float32",
            "frame_joint_matrices_shape": [skeleton["joint_count"], 4, 4],
            "clip_joint_matrices_shape": [len(frames), skeleton["joint_count"], 4, 4],
        })
        for index, matrices in enumerate(exported_animation_skeleton):
            np.save(skeleton_dir / f"frame_{index:04d}.npy", matrices)
    # Recreate the exact state used to capture reference_vertices. For assets
    # without an armature, object actions remain active in REST mode, so using
    # the clip's first frame here would apply its object translation twice.
    scene.frame_set(reference_source_frame)
    set_armature_pose_position(scene, "REST")
    originals = []
    try:
        originals = apply_reference_transform(scene, reference_transform)
        aligned_reference, aligned_faces, aligned_topology = evaluated_geometry(meshes, depsgraph)
        if aligned_topology != topology or not np.array_equal(aligned_faces, faces):
            raise RuntimeError(f"Topology changed while aligning reference for clip {clip_index}")
        # Stateful modifiers can drift across repeated depsgraph evaluations.
        # Center the exact snapshot being written, and apply the same final
        # translation to the scene without evaluating the exported array again.
        reference_output_transform = np.eye(4, dtype=np.float64)
        reference_output_transform[:3, 3] = origin_centering_translation(aligned_reference)
        aligned_reference = transform_vertices(aligned_reference, reference_output_transform)
        reference_transform = Matrix(reference_output_transform.tolist()) @ reference_transform
        replace_applied_transform(originals, reference_transform)
        reference_centroid_array = require_origin_centroid(
            f"clip {clip_index} reference", aligned_reference
        )
        reference_alignment["reference_transform"] = matrix_to_list(reference_transform)
        reference_alignment["exported_reference_centroid"] = (
            reference_centroid_array.astype(float).tolist()
        )
        reference_alignment["origin_centroid_tolerance"] = ORIGIN_CENTROID_TOLERANCE
        reference_alignment["output_centering_translation"] = (
            reference_output_transform[:3, 3].astype(float).tolist()
        )
        np.savez(staging / "reference_mesh.npz", vertices=aligned_reference, faces=faces)
        reference_bounds_vertices = np.stack(
            (aligned_reference.min(axis=0), aligned_reference.max(axis=0))
        )
        reference_camera_metadata = update_camera(
            scene, target_camera, target_lights, reference_bounds_vertices, camera_view(0), lighting_preset
        )
        reference_camera = build_reference_camera(aligned_reference, reference_camera_metadata)
        scene.render.filepath = str(reference_dir / "camera_0.png")
        bpy.ops.render.render(write_still=True)
    finally:
        if originals:
            restore_reference_transform(originals)
        set_armature_pose_position(scene, "POSE")

    reference_foreground = measure_reference_foreground(
        reference_dir / "camera_0.png", args.reference_background_pixel_threshold
    )
    if reference_foreground["foreground_pixel_fraction"] <= args.min_reference_foreground:
        shutil.rmtree(staging)
        return {
            "status": "skipped_low_reference_foreground",
            "clip_index": clip_index,
            "frames": len(frames),
            "motion": motion,
            "reference_foreground": reference_foreground,
            "reference_foreground_filter": {
                "min_foreground_pixel_fraction": args.min_reference_foreground,
                "passed": False,
            },
        }

    reference_centroid = centroid64(aligned_reference).astype(float).tolist()
    rows, frame_metadata_by_camera, image_paths_by_camera = [], {}, {}
    frame_metadata = []
    vertex_files = []
    skeleton_files = []
    for index, (source_frame, vertices) in enumerate(zip(frames, exported_animation_vertices)):
        scene.frame_set(source_frame)
        bpy.context.view_layer.update()
        vertex_file = f"frame_{index:04d}.npy"
        skeleton_file = f"frame_{index:04d}.npy" if skeleton else None
        np.save(vertices_dir / vertex_file, vertices)
        vertex_files.append(vertex_file)
        skeleton_files.append(skeleton_file)

    target_originals = apply_reference_transform(scene, Matrix(target_translation.tolist()))
    try:
        for camera_index in sampled_indices:
            view = camera_view(camera_index)
            camera_name = view["name"]
            camera_dir = images_dir / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            update_camera(scene, target_camera, target_lights, bounds_vertices, view, lighting_preset)
            camera_metadata = target_camera_metadata_by_name[camera_name]
            camera_frames = []
            image_paths = []
            for index, (source_frame, vertices) in enumerate(zip(frames, exported_animation_vertices)):
                scene.frame_set(source_frame)
                bpy.context.view_layer.update()
                image_file = f"frame_{index:04d}.png"
                camera_data = build_frame_camera(vertices, camera_metadata)
                scene.render.filepath = str(camera_dir / image_file)
                bpy.ops.render.render(write_still=True)
                image_paths.append(camera_dir / image_file)
                camera_frames.append({"frame_index": index, "source_frame": source_frame, **camera_data})
            frame_metadata_by_camera[camera_name] = camera_frames
            image_paths_by_camera[camera_name] = image_paths
    finally:
        restore_reference_transform(target_originals)

    frame_metadata = frame_metadata_by_camera["camera_0"]
    image_change_by_camera = {
        camera_name: measure_image_change(paths, args.image_change_pixel_threshold)
        for camera_name, paths in image_paths_by_camera.items()
    }
    image_change = image_change_by_camera["camera_0"]
    if image_change["mean_changed_pixel_fraction_per_frame"] <= args.min_image_change:
        shutil.rmtree(staging)
        return {
            "status": "skipped_low_image_change",
            "clip_index": clip_index,
            "frames": len(frames),
            "motion": motion,
            "image_change": image_change,
            "image_change_filter": {
                "min_mean_changed_pixel_fraction_per_frame": args.min_image_change,
                "passed": False,
            },
        }
    for camera_index in sampled_indices:
        camera_name = camera_view(camera_index)["name"]
        for index, (source_frame, vertices) in enumerate(zip(frames, exported_animation_vertices)):
            rows.append(build_sample_row(
                args=args,
                clip_index=clip_index,
                frame_index=index,
                source_frame=source_frame,
                final_root=manifest_root,
                reference_centroid=reference_centroid,
                frame_centroid=centroid64(vertices).astype(float).tolist(),
                motion=motion,
                vertex_file=vertex_files[index],
                image_file=f"frame_{index:04d}.png",
                camera_index=camera_index,
                camera_name=camera_name,
                skeleton_file=skeleton_files[index],
            ))

    for row in rows:
        row_image_change = image_change_by_camera[row["camera_name"]]
        row["clip_mean_changed_pixel_fraction_per_frame"] = row_image_change[
            "mean_changed_pixel_fraction_per_frame"
        ]
        row["reference_foreground_pixel_fraction"] = reference_foreground[
            "foreground_pixel_fraction"
        ]

    source_manifest = build_clip_manifest(
        common=clip_common,
        clip_index=clip_index,
        frames=frames,
        motion=motion,
        min_motion=args.min_motion,
        image_change=image_change,
        min_image_change=args.min_image_change,
        reference_foreground=reference_foreground,
        min_reference_foreground=args.min_reference_foreground,
        aligned_reference=aligned_reference,
        faces=faces,
        final_root=manifest_root,
        reference_alignment=reference_alignment,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        reference_camera=reference_camera,
        target_camera=target_camera_metadata,
        frame_metadata=frame_metadata,
        target_cameras=target_camera_metadata_by_name,
        sampled_camera_indices=sampled_indices,
        frame_metadata_by_camera=frame_metadata_by_camera,
        image_change_by_camera=image_change_by_camera,
    )
    write_json(staging / "source_manifest.json", source_manifest)
    write_json(staging / "cameras.json", {
        "reference": {
            "camera_0": reference_camera,
            "mode": "reference_from_reference_mesh_bbox",
        },
        "target": {
            "camera_0": target_camera_metadata,
            "views": target_camera_metadata_by_name,
            "sampled_camera_indices": sampled_indices,
            "mode": "fixed_from_clip_animation_union_bounds",
            "frames": frame_metadata_by_camera,
        },
    })
    write_jsonl(staging / "sample_manifest.jsonl", rows)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        shutil.rmtree(final_root)
    staging.replace(final_root)
    return {
        "status": "complete",
        "clip_index": clip_index,
        "frames": len(frames),
        "motion": motion,
        "image_change": image_change,
        "image_change_by_camera": image_change_by_camera,
        "reference_foreground": reference_foreground,
        "output": str(final_root),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_image_change <= 1.0:
        raise ValueError("--min-image-change must be between 0 and 1")
    if args.max_centroid_motion_bbox_ratio < 0.0:
        raise ValueError("--max-centroid-motion-bbox-ratio must be non-negative")
    if not 0 <= args.image_change_pixel_threshold <= 255:
        raise ValueError("--image-change-pixel-threshold must be between 0 and 255")
    if not 0.0 <= args.min_reference_foreground <= 1.0:
        raise ValueError("--min-reference-foreground must be between 0 and 1")
    if not 0 <= args.reference_background_pixel_threshold <= 255:
        raise ValueError("--reference-background-pixel-threshold must be between 0 and 255")
    imported_scene(args.source)
    scene = bpy.context.scene
    reference_source_frame = int(scene.frame_current)
    meshes, mesh_filter = select_meshes(scene)
    forced_opaque_materials = force_opaque_materials(meshes)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    armatures = set_armature_pose_position(scene, "REST")
    skeleton = skeleton_descriptor(scene, meshes)
    reference_source, faces, topology = evaluated_geometry(meshes, depsgraph)
    normalization = normalize_import(scene, meshes, reference_source.min(axis=0), reference_source.max(axis=0))
    reference_vertices, reference_faces, reference_topology = evaluated_geometry(meshes, depsgraph)
    if reference_topology != topology or not np.array_equal(reference_faces, faces):
        raise RuntimeError("Topology changed while applying reference normalization")
    effective_start, effective_end, effective_range = effective_animation_range(scene, meshes)
    candidates, fps, sampling = sampled_frames(scene, args.max_fps, effective_start, effective_end)
    clips, clip_selection = split_clips(candidates, args.sample_id, args.clip_frames, args.max_clips)
    raw_destination = args.raw_root / Path(args.source_archive).parent / args.sample_id
    common = build_common_metadata(
        args=args,
        fps=fps,
        sampling=sampling,
        effective_range=effective_range,
        clip_selection=clip_selection,
        raw_destination=raw_destination,
        forced_opaque_materials=forced_opaque_materials,
        mesh_filter=mesh_filter,
        topology=topology,
        armatures=armatures,
        reference_source_frame=reference_source_frame,
        normalization=normalization,
        skeleton=skeleton,
    )
    # A corrupt/topology-changing clip must not discard other valid clips from
    # the same animation. Each clip is staged and published independently.
    results = []
    for index, frames in enumerate(clips, start=1):
        try:
            results.append(export_clip(
                args, scene, meshes, depsgraph, faces, topology,
                reference_vertices, reference_source_frame, common, index, frames,
            ))
        except Exception as error:
            results.append({
                "status": "failed",
                "clip_index": index,
                "frames": len(frames),
                "error": str(error),
                "traceback": traceback.format_exc(),
            })
    completed = [value for value in results if value["status"] == "complete"]
    failed = [value for value in results if value["status"] == "failed"]
    if completed:
        status = "complete"
    elif failed:
        status = "failed_no_valid_clips"
    else:
        status = "skipped_no_valid_clips"
    print("TEXVERSE_RESULT=" + json.dumps({
        "status": status,
        "sample_id": args.sample_id,
        "clip_count": len(completed),
        "candidate_clip_count": len(clips),
        "clips": results,
    }), flush=True)


if __name__ == "__main__":
    main()
