#!/usr/bin/env python3
"""Export one normalized TexVerse FBX/GLB asset into LTA-style mesh frames.

Run with Blender. The asset is normalized as a whole, but its reference and
animated mesh centroids are recorded rather than root-aligned.
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

from pipeline_logic import PIPELINE_VERSION, snap_rotation_to_axis_aligned, split_clips


NORMALIZED_MAX_EXTENT = 2.0


def matrix_to_list(matrix: Matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--max-fps", type=float, default=16.0)
    parser.add_argument("--min-motion", type=float, default=0.01)
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
        raise RuntimeError("Scene-mesh filtering removed every renderable mesh")
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


def root_joint_alignment(scene, meshes: list) -> tuple[Matrix, dict]:
    """Build the rigid REST-to-first-POSE transform for one deterministic root bone."""
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
    root = roots[0]
    # Called in REST mode first, then again in POSE mode by the caller.
    rest_matrix = armature.matrix_world @ root.matrix_local
    return rest_matrix, {"applied": True, "armature": armature.name, "bone": root.name}


def root_joint_pose_matrix(scene, armature_name: str, bone_name: str) -> Matrix:
    armature = scene.objects[armature_name]
    pose_bone = armature.pose.bones[bone_name]
    return armature.matrix_world @ pose_bone.matrix


def canonical_reference_transform(
    scene,
    meshes: list,
    reference_vertices: np.ndarray,
    target_centroid: np.ndarray,
) -> tuple[Matrix, dict]:
    """Align REST to the nearest 3D axis-aligned rotation of the first pose."""
    rest_matrix, root_metadata = root_joint_alignment(scene, meshes)
    reference_centroid = reference_vertices.mean(axis=0)
    scale_factor = 1.0
    rest_rotation = first_target_rotation = canonical_target_rotation = None
    canonical_rotation_index = None
    canonical_target_axes = None

    if root_metadata.get("applied") is False:
        rotation = Matrix.Identity(4)
        method = "translation_only_no_armature"
        pose_matrix = None
    else:
        pose_matrix = root_joint_pose_matrix(scene, root_metadata["armature"], root_metadata["bone"])
        rest_scale = np.asarray(rest_matrix.to_scale(), dtype=np.float32)
        pose_scale = np.asarray(pose_matrix.to_scale(), dtype=np.float32)
        ratios = pose_scale / np.maximum(rest_scale, 1e-8)
        # Use the root's uniform scale, or its geometric-average equivalent for noisy FBX matrices.
        scale_factor = float(np.exp(np.log(np.maximum(ratios, 1e-8)).mean()))

        rest_rotation = rest_matrix.to_quaternion().to_matrix()
        first_target_rotation = pose_matrix.to_quaternion().to_matrix()
        snapped, canonical_rotation_index, canonical_target_axes = snap_rotation_to_axis_aligned(
            first_target_rotation
        )
        canonical_target_rotation = Matrix(snapped)
        rotation = (canonical_target_rotation @ rest_rotation.inverted()).to_4x4()
        method = "align_rest_root_to_nearest_3d_axis_rotation_of_first_target_root_then_match_centroid"

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
        "first_target_root_rotation": (
            matrix_to_list(first_target_rotation) if first_target_rotation is not None else None
        ),
        "canonical_target_rotation": (
            matrix_to_list(canonical_target_rotation) if canonical_target_rotation is not None else None
        ),
        "canonical_target_axes": list(canonical_target_axes) if canonical_target_axes is not None else None,
        "canonical_rotation_index": canonical_rotation_index,
        "applied_rotation": matrix_to_list(rotation),
        "reference_uniform_scale": scale_factor,
        "root_rest_scale": [float(value) for value in rest_matrix.to_scale()],
        "root_first_target_scale": (
            [float(value) for value in pose_matrix.to_scale()] if pose_matrix is not None else None
        ),
        "reference_centroid_before": reference_centroid.astype(float).tolist(),
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


def measure_motion(meshes: list, depsgraph, topology: list[dict], frames: list[int], vertex_indices: np.ndarray) -> dict:
    previous_vertices = previous_centroid = None
    relative_steps: list[float] = []
    global_centroid_steps: list[float] = []
    for source_frame in frames:
        bpy.context.scene.frame_set(source_frame)
        bpy.context.view_layer.update()
        current_vertices, current_centroid = evaluated_motion_state(meshes, depsgraph, topology, vertex_indices)
        if previous_vertices is not None:
            # Remove whole-mesh translation before assessing deformation / articulated motion.
            previous_relative = previous_vertices - previous_centroid
            current_relative = current_vertices - current_centroid
            relative_steps.append(float(np.linalg.norm(current_relative - previous_relative, axis=1).mean()))
            global_centroid_steps.append(float(np.linalg.norm(current_centroid - previous_centroid)))
        previous_vertices, previous_centroid = current_vertices, current_centroid
    return {
        "method": "mean_euclidean_displacement_of_deterministic_vertex_sample_after_subtracting_each_frame_full_mesh_centroid",
        "frame_definition": "consecutive_exported_frames_after_fps_downsampling_and_frame_window_selection",
        "vertex_sample_count": int(len(vertex_indices)),
        "vertex_indices": vertex_indices.astype(int).tolist(),
        "step_count": len(relative_steps),
        "mean_per_point_per_frame": float(np.mean(relative_steps)) if relative_steps else 0.0,
        "step_mean_displacements": relative_steps,
        "global_centroid_motion_mean_per_frame": float(np.mean(global_centroid_steps)) if global_centroid_steps else 0.0,
        "global_centroid_step_displacements": global_centroid_steps,
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


FRONT_VIEW = {
    "name": "front_negative_y",
    "offset": np.asarray((0.0, -1.0, 0.0)),
    "horizontal": 0,
    "vertical": 2,
    "depth": 1,
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
    view_direction: str,
) -> float:
    """Fit a perspective camera to projected bounds with five percent empty space per edge."""
    half_extent = 0.5 * (maximum - minimum)
    view = FRONT_VIEW
    usable_half_image = 0.90  # Five percent empty space on each image edge.
    horizontal_tangent, vertical_tangent = effective_camera_half_tangents(scene, camera_data)
    horizontal = half_extent[view["horizontal"]] / max(horizontal_tangent * usable_half_image, 1e-6)
    vertical = half_extent[view["vertical"]] / max(vertical_tangent * usable_half_image, 1e-6)
    # The nearest depth bound, rather than the mesh centre, controls perspective cropping.
    return float(half_extent[view["depth"]] + max(horizontal, vertical, 0.001))


def configure_render(scene, resolution: int, vertices: np.ndarray, render_threads: int):
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
    view_direction = FRONT_VIEW["name"]
    distance = camera_distance_for_bounds(scene, camera_data, minimum, maximum, view_direction)
    camera.location = tuple(center + FRONT_VIEW["offset"] * distance)
    camera.rotation_euler = (Vector(center) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.clip_start = 0.001
    camera_data.clip_end = 100.0
    scene.camera = camera
    lights = []
    for name, position, energy in (("Key", center + np.asarray((-2, -2, 3)), 1200.0), ("Fill", center + np.asarray((2, -1, 1)), 600.0)):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = 4.0
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light["texverse_export_helper"] = True
        light.location = tuple(position)
        light.rotation_euler = (Vector(center) - light.location).to_track_quat("-Z", "Y").to_euler()
        lights.append((light, np.asarray((-2, -2, 3)) if name == "Key" else np.asarray((2, -1, 1))))
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
    background.inputs["Color"].default_value = (0.15, 0.15, 0.15, 1.0)
    background.inputs["Strength"].default_value = 0.35
    return camera, lights, view_direction


def remove_export_render_helpers(scene) -> None:
    """Drop a prior clip's camera/lights so clip illumination never accumulates."""
    for obj in list(scene.objects):
        if obj.get("texverse_export_helper"):
            bpy.data.objects.remove(obj, do_unlink=True)


def update_camera(scene, camera, lights: list, vertices: np.ndarray, view_direction: str) -> dict:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    distance = camera_distance_for_bounds(scene, camera.data, minimum, maximum, view_direction)
    camera.location = tuple(center + FRONT_VIEW["offset"] * distance)
    camera.rotation_euler = (Vector(center) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.clip_end = max(100.0, distance * 4.0)
    # Keep illumination relative to each independently framed mesh pose.
    for light, offset in lights:
        light.location = tuple(center + offset)
        light.rotation_euler = (Vector(center) - light.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    return {
        "centroid": vertices.mean(axis=0).astype(float).tolist(),
        "bounds_min": minimum.astype(float).tolist(),
        "bounds_max": maximum.astype(float).tolist(),
        "camera_location": list(camera.location),
        "camera_rotation_euler": list(camera.rotation_euler),
        "camera_lens_mm": float(camera.data.lens),
        "view_direction": view_direction,
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
        "camera_mode": "shared_with_clip_animation_union_bounds",
        "reference_subject": bounds_metadata(aligned_reference),
    }


def build_frame_camera(vertices: np.ndarray, target_camera: dict, view_direction: str) -> dict:
    return {
        **bounds_metadata(vertices),
        "camera_location": target_camera["camera_location"],
        "camera_rotation_euler": target_camera["camera_rotation_euler"],
        "camera_lens_mm": target_camera["camera_lens_mm"],
        "view_direction": view_direction,
        "camera_mode": "fixed_from_clip_animation_union_bounds",
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
) -> dict:
    return {
        "sample_id": args.sample_id,
        "clip_index": clip_index,
        "split": "train",
        "frame_index": frame_index,
        "source_frame": source_frame,
        "view_label": "front",
        "sample_root": str(final_root),
        "source_manifest": str(final_root / "source_manifest.json"),
        "rest_mesh": str(final_root / "reference_mesh.npz"),
        "target_vertices": str(final_root / "target_vertices" / vertex_file),
        "target_image": str(final_root / "target_images" / image_file),
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
            "reference_camera_mode": "shared_with_clip_animation_union_bounds",
            "target_camera_mode": "fixed_from_clip_animation_union_bounds",
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
    }


def build_clip_manifest(
    common: dict,
    clip_index: int,
    frames: list[int],
    motion: dict,
    min_motion: float,
    aligned_reference: np.ndarray,
    faces: np.ndarray,
    final_root: Path,
    reference_alignment: dict,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    reference_camera: dict,
    target_camera: dict,
    frame_metadata: list[dict],
) -> dict:
    return {
        **common,
        "clip_index": clip_index,
        "exported_frame_count": len(frames),
        "exported_source_frames": frames,
        "motion": motion,
        "motion_filter": {"min_motion": min_motion, "passed": True},
        "vertex_count": int(len(aligned_reference)),
        "face_count": int(len(faces)),
        "reference_mesh": str(final_root / "reference_mesh.npz"),
        "reference_image": str(final_root / "reference_views" / "front.png"),
        "reference_centroid": aligned_reference.mean(axis=0).astype(float).tolist(),
        "reference_alignment": reference_alignment,
        "animation_union_bounds": {
            "min": bounds_min.astype(float).tolist(),
            "max": bounds_max.astype(float).tolist(),
        },
        "reference_camera": reference_camera,
        "target_camera": target_camera,
        "frames": frame_metadata,
    }


def export_clip(
    args, scene, meshes, depsgraph, faces, topology, reference_vertices,
    reference_source_frame, common, clip_index, frames,
):
    """Validate and export one independent clip under <animation-id>/<clip-index>."""
    remove_export_render_helpers(scene)
    set_armature_pose_position(scene, "POSE")
    animation_vertices = []
    for source_frame in frames:
        scene.frame_set(source_frame)
        bpy.context.view_layer.update()
        vertices, frame_faces, current_topology = evaluated_geometry(meshes, depsgraph)
        if current_topology != topology or not np.array_equal(frame_faces, faces):
            raise RuntimeError(f"Topology changed in clip {clip_index} at frame {source_frame}")
        animation_vertices.append(vertices)
    rng = np.random.default_rng(int(hashlib.sha256(f"{args.sample_id}:{clip_index}".encode()).hexdigest()[:16], 16))
    indices = np.sort(rng.choice(len(reference_vertices), size=min(100, len(reference_vertices)), replace=False))
    motion = measure_motion(meshes, depsgraph, topology, frames, indices)
    if motion["mean_per_point_per_frame"] <= args.min_motion:
        return {"status": "skipped_low_motion", "clip_index": clip_index, "motion": motion, "frames": len(frames)}
    first_vertices = animation_vertices[0]
    # measure_motion leaves Blender evaluated at the final frame. Reference
    # scale/alignment must instead use this clip's first animated frame.
    scene.frame_set(frames[0])
    bpy.context.view_layer.update()
    reference_transform, reference_alignment = canonical_reference_transform(
        scene, meshes, reference_vertices, first_vertices.mean(axis=0)
    )
    reference_alignment["first_target_source_frame"] = frames[0]
    bounds_min = np.min([vertices.min(axis=0) for vertices in animation_vertices], axis=0)
    bounds_max = np.max([vertices.max(axis=0) for vertices in animation_vertices], axis=0)
    bounds_vertices = np.stack((bounds_min, bounds_max))
    target_camera, target_lights, view_direction = configure_render(scene, args.resolution, bounds_vertices, args.render_threads)
    target_camera_metadata = update_camera(scene, target_camera, target_lights, bounds_vertices, view_direction)
    final_root = args.output_root / Path(args.source_archive).parent / args.sample_id / str(clip_index)
    staging = args.output_root / ".staging" / Path(args.source_archive).parent / args.sample_id / str(clip_index)
    if staging.exists():
        shutil.rmtree(staging)
    vertices_dir, images_dir, reference_dir = staging / "target_vertices", staging / "target_images", staging / "reference_views"
    for directory in (vertices_dir, images_dir, reference_dir):
        directory.mkdir(parents=True, exist_ok=True)
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
        np.savez(staging / "reference_mesh.npz", vertices=aligned_reference, faces=faces)
        reference_camera = build_reference_camera(aligned_reference, target_camera_metadata)
        scene.render.filepath = str(reference_dir / "front.png")
        bpy.ops.render.render(write_still=True)
    finally:
        if originals:
            restore_reference_transform(originals)
        set_armature_pose_position(scene, "POSE")

    reference_centroid = aligned_reference.mean(axis=0).astype(float).tolist()
    rows, frame_metadata = [], []
    for index, (source_frame, vertices) in enumerate(zip(frames, animation_vertices)):
        scene.frame_set(source_frame)
        bpy.context.view_layer.update()
        vertex_file, image_file = f"frame_{index:04d}.npy", f"frame_{index:04d}.png"
        np.save(vertices_dir / vertex_file, vertices)
        camera_data = build_frame_camera(vertices, target_camera_metadata, view_direction)
        scene.render.filepath = str(images_dir / image_file)
        bpy.ops.render.render(write_still=True)
        frame_metadata.append({"frame_index": index, "source_frame": source_frame, **camera_data})
        rows.append(build_sample_row(
            args=args,
            clip_index=clip_index,
            frame_index=index,
            source_frame=source_frame,
            final_root=final_root,
            reference_centroid=reference_centroid,
            frame_centroid=vertices.mean(axis=0).astype(float).tolist(),
            motion=motion,
            vertex_file=vertex_file,
            image_file=image_file,
        ))

    source_manifest = build_clip_manifest(
        common=common,
        clip_index=clip_index,
        frames=frames,
        motion=motion,
        min_motion=args.min_motion,
        aligned_reference=aligned_reference,
        faces=faces,
        final_root=final_root,
        reference_alignment=reference_alignment,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        reference_camera=reference_camera,
        target_camera=target_camera_metadata,
        frame_metadata=frame_metadata,
    )
    write_json(staging / "source_manifest.json", source_manifest)
    write_json(staging / "cameras.json", {
        "reference": {
            "front": reference_camera,
            "mode": "shared_with_clip_animation_union_bounds",
        },
        "target": {
            "front": target_camera_metadata,
            "mode": "fixed_from_clip_animation_union_bounds",
            "frames": frame_metadata,
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
        "output": str(final_root),
    }


def main() -> None:
    args = parse_args()
    imported_scene(args.source)
    scene = bpy.context.scene
    reference_source_frame = int(scene.frame_current)
    meshes, mesh_filter = select_meshes(scene)
    forced_opaque_materials = force_opaque_materials(meshes)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    armatures = set_armature_pose_position(scene, "REST")
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
