#!/usr/bin/env python3
"""
Blender to Caramel Scene Converter

Usage (from command line):
    blender scene.blend --background --python blend_to_caramel.py -- output_dir [--spp 200] [--depth-max 5] [--depth-rr 0] [--width 800] [--height 600]

Usage (inside Blender scripting tab):
    Modify OUTPUT_DIR below and run the script.

Converts a Blender scene into a Caramel-compatible JSON scene file with exported OBJ meshes.
"""

import bpy
import json
import os
import sys
import math
import argparse
from mathutils import Vector, Matrix


# ============================================================
# Configuration defaults (overridable via CLI args)
# ============================================================
DEFAULT_SPP = 200
DEFAULT_DEPTH_MAX = 5
DEFAULT_DEPTH_RR = 0
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600

# Fallback output directory when run from Blender GUI
OUTPUT_DIR = "/tmp/caramel_export"


def parse_args():
    """Parse arguments after '--' in blender CLI invocation."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        return None

    parser = argparse.ArgumentParser(description="Export Blender scene to Caramel format")
    parser.add_argument("output_dir", help="Directory to write scene.json and meshes/")
    parser.add_argument("--spp", type=int, default=DEFAULT_SPP)
    parser.add_argument("--depth-max", type=int, default=DEFAULT_DEPTH_MAX)
    parser.add_argument("--depth-rr", type=int, default=DEFAULT_DEPTH_RR)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-split-objects", action="store_true", default=False,
                        help="Export as single OBJ without splitting by 'o' groups. "
                             "Requires multi-shape OBJ support in Caramel.")
    return parser.parse_args(argv)


# ============================================================
# Helpers
# ============================================================

def vec3_to_list(v):
    """Convert a Blender Vector/tuple to [x, y, z] list of floats."""
    return [round(float(v[0]), 6), round(float(v[1]), 6), round(float(v[2]), 6)]


def blender_to_caramel_vec(v):
    """Convert a Blender Z-up vector to Caramel Y-up: (x, y, z) -> (x, z, -y)."""
    return [round(float(v[0]), 6), round(float(v[2]), 6), round(float(-v[1]), 6)]


def color_to_list(c):
    """Convert a Blender color (3 or 4 channels) to [r, g, b]."""
    return [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)]


def decompose_transform(obj):
    """Convert an object's world transform to a Caramel to_world 4x4 matrix.
    Uses matrix_world to correctly handle parent hierarchies.
    Converts Blender Z-up coordinates to Caramel Y-up coordinates."""
    from mathutils import Matrix as BlMatrix

    mw = obj.matrix_world

    # Check if it's effectively identity
    identity = BlMatrix.Identity(4)
    is_identity = all(
        math.isclose(mw[r][c], identity[r][c], abs_tol=1e-5)
        for r in range(4) for c in range(4)
    )
    if is_identity:
        return None

    # Coordinate conversion: Blender Z-up -> Caramel Y-up
    # (x, y, z) -> (x, z, -y)
    # M_caramel = C @ M_blender @ C_inv
    # where C = [[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]
    # and C_inv = [[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]
    coord_conv = BlMatrix((
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    ))
    coord_inv = BlMatrix((
        (1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    ))

    m = coord_conv @ mw @ coord_inv

    # Output as 16-element row-major array for Caramel
    # Blender Matrix4x4 is row-major accessible via m[row][col]
    return [round(float(m[r][c]), 6) for r in range(4) for c in range(4)]


def get_principled_bsdf_node(material):
    """Find the Principled BSDF node in a material's node tree."""
    if not material or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    return None


def get_diffuse_bsdf_node(material):
    """Find a Diffuse BSDF node in a material's node tree."""
    if not material or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == 'BSDF_DIFFUSE':
            return node
    return None


def get_glossy_bsdf_node(material):
    """Find a Glossy BSDF node in a material's node tree."""
    if not material or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == 'BSDF_GLOSSY':
            return node
    return None


def get_glass_bsdf_node(material):
    """Find a Glass BSDF node in a material's node tree."""
    if not material or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == 'BSDF_GLASS':
            return node
    return None


def get_image_texture_node(socket):
    """Follow a socket's links to find a connected Image Texture node.
    Recursively traverses intermediate nodes (Mix, ColorRamp, HueSat, etc.)."""
    if not socket.is_linked:
        return None
    linked_node = socket.links[0].from_node
    if linked_node.type == 'TEX_IMAGE' and linked_node.image:
        return linked_node
    # Recurse through intermediate nodes
    for inp in linked_node.inputs:
        if inp.type in ('RGBA', 'VALUE', 'VECTOR') and inp.is_linked:
            result = get_image_texture_node(inp)
            if result:
                return result
    return None


def _has_complex_shader_tree(socket, depth=0):
    """Check if a socket's node tree involves procedural textures or Mix nodes
    that cannot be represented by a single image texture."""
    if not socket.is_linked or depth > 10:
        return False
    linked_node = socket.links[0].from_node
    if linked_node.type == 'TEX_IMAGE':
        return False
    # Procedural textures, node groups, or complex color operations → needs bake
    if linked_node.type in ('TEX_NOISE', 'TEX_VORONOI', 'TEX_GRADIENT',
                            'TEX_MUSGRAVE', 'TEX_WAVE', 'TEX_MAGIC',
                            'TEX_CHECKER', 'TEX_BRICK', 'GROUP'):
        return True
    if linked_node.type in ('MIX', 'MIX_SHADER', 'VALTORGB'):
        # Check if any input leads to procedural or multiple textures
        for inp in linked_node.inputs:
            if inp.is_linked and _has_complex_shader_tree(inp, depth + 1):
                return True
    for inp in linked_node.inputs:
        if inp.is_linked and _has_complex_shader_tree(inp, depth + 1):
            return True
    return False


def _bake_material_to_texture(obj, material, output_dir, bake_size=1024):
    """Bake a material's Base Color to an image texture using Blender's bake system.
    Returns the relative path to the baked texture, or None on failure."""
    tex_dir = os.path.join(output_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)

    mat_name = bpy.path.clean_name(material.name)
    dst_name = f"{mat_name}_baked.png"
    dst_path = os.path.join(tex_dir, dst_name)

    # Check if the mesh has UV coordinates
    if not obj.data.uv_layers:
        print(f"  WARNING: '{obj.name}' has no UV map, cannot bake texture for '{material.name}'")
        return None

    # Save current state
    orig_engine = bpy.context.scene.render.engine
    orig_selected = [o for o in bpy.context.selected_objects]
    orig_active = bpy.context.view_layer.objects.active

    try:
        bpy.context.scene.render.engine = 'CYCLES'

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Create a temporary image for baking
        bake_img = bpy.data.images.new(
            name=f"_bake_{mat_name}", width=bake_size, height=bake_size, alpha=False
        )

        # Use the original (non-evaluated) material for node tree manipulation
        orig_material = bpy.data.materials.get(material.name, material)
        node_tree = orig_material.node_tree
        bake_node = node_tree.nodes.new('ShaderNodeTexImage')
        bake_node.image = bake_img
        bake_node.name = "_bake_target"
        # Must be selected (active) node for bake
        node_tree.nodes.active = bake_node

        # Ensure the correct material is assigned
        mat_index = None
        for i, slot in enumerate(obj.material_slots):
            if slot.material and slot.material.name == material.name:
                mat_index = i
                break
        if mat_index is None:
            print(f"  WARNING: Material '{material.name}' not found on '{obj.name}'")
            node_tree.nodes.remove(bake_node)
            bpy.data.images.remove(bake_img)
            return None

        # Bake diffuse color (no direct/indirect lighting contributions)
        bpy.context.scene.cycles.bake_type = 'DIFFUSE'
        bpy.context.scene.render.bake.use_pass_direct = False
        bpy.context.scene.render.bake.use_pass_indirect = False
        bpy.context.scene.render.bake.use_pass_color = True

        bpy.ops.object.bake(type='DIFFUSE')

        # Save the baked image
        bake_img.filepath_raw = dst_path
        bake_img.file_format = 'PNG'
        bake_img.save()

        print(f"  Baked texture for '{material.name}' -> {dst_name}")

        # Cleanup
        node_tree.nodes.remove(bake_node)
        bpy.data.images.remove(bake_img)

        return os.path.join("textures", dst_name)

    except Exception as e:
        print(f"  WARNING: Failed to bake texture for '{material.name}': {e}")
        # Cleanup on failure
        orig_mat = bpy.data.materials.get(material.name, material)
        if "_bake_target" in orig_mat.node_tree.nodes:
            orig_mat.node_tree.nodes.remove(orig_mat.node_tree.nodes["_bake_target"])
        if f"_bake_{mat_name}" in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[f"_bake_{mat_name}"])
        return None

    finally:
        # Restore state
        bpy.context.scene.render.engine = orig_engine
        bpy.ops.object.select_all(action='DESELECT')
        for o in orig_selected:
            o.select_set(True)
        bpy.context.view_layer.objects.active = orig_active


def _export_texture_image(image, output_dir):
    """Save a Blender image to the output textures directory, return relative path.
    Handles external files, packed (embedded) images, and generated images."""
    tex_dir = os.path.join(output_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)

    # Supported formats in Caramel
    SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.exr', '.hdr'}

    # 1) Try to copy the original file if it exists on disk
    src_path = bpy.path.abspath(image.filepath) if image.filepath else ""
    if src_path and os.path.exists(src_path):
        ext = os.path.splitext(src_path)[1].lower()
        if ext in SUPPORTED_EXTS:
            import shutil
            dst_name = os.path.basename(src_path)
            dst_path = os.path.join(tex_dir, dst_name)
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                shutil.copy2(src_path, dst_path)
            return os.path.join("textures", dst_name)
        # Unsupported format — convert to PNG via Blender
        dst_name = os.path.splitext(os.path.basename(src_path))[0] + ".png"
        dst_path = os.path.join(tex_dir, dst_name)
        image.save_render(filepath=dst_path)
        print(f"  Converted {ext} texture to PNG: {dst_name}")
        return os.path.join("textures", dst_name)

    # 2) Packed image — extract embedded data from .blend
    if image.packed_file:
        ext = os.path.splitext(os.path.basename(image.filepath))[1].lower() if image.filepath else ""
        if not ext:
            fmt_map = {'PNG': '.png', 'JPEG': '.jpg', 'BMP': '.bmp',
                       'TARGA': '.tga', 'OPEN_EXR': '.exr', 'HDR': '.hdr'}
            ext = fmt_map.get(image.file_format, '.png')
        if ext in SUPPORTED_EXTS:
            dst_name = bpy.path.clean_name(image.name) + ext
            dst_path = os.path.join(tex_dir, dst_name)
            with open(dst_path, 'wb') as f:
                f.write(image.packed_file.data)
            print(f"  Extracted packed texture: {dst_name}")
            return os.path.join("textures", dst_name)
        # Unsupported packed format — convert to PNG via Blender
        dst_name = bpy.path.clean_name(image.name) + ".png"
        dst_path = os.path.join(tex_dir, dst_name)
        image.save_render(filepath=dst_path)
        print(f"  Converted packed {ext} texture to PNG: {dst_name}")
        return os.path.join("textures", dst_name)

    # 3) Generated or render-result image — save via Blender API
    dst_name = bpy.path.clean_name(image.name) + ".png"
    dst_path = os.path.join(tex_dir, dst_name)
    image.save_render(filepath=dst_path)
    print(f"  Saved generated texture: {dst_name}")
    return os.path.join("textures", dst_name)


def convert_material(material, output_dir, obj=None):
    """
    Convert a Blender material to a Caramel BSDF dict.
    Supports diffuse with color, image texture, or baked procedural texture.
    """
    if not material:
        return {"type": "diffuse"}

    # Try to extract base color from Principled BSDF
    node = get_principled_bsdf_node(material)
    if node:
        base_color_input = node.inputs['Base Color']
        tex_node = get_image_texture_node(base_color_input)
        if tex_node:
            rel_path = _export_texture_image(tex_node.image, output_dir)
            return {"type": "diffuse", "texture": {"type": "image", "path": rel_path}}
        # Complex shader tree (procedural, Mix, ColorRamp, etc.) → bake
        if obj and base_color_input.is_linked and _has_complex_shader_tree(base_color_input):
            rel_path = _bake_material_to_texture(obj, material, output_dir)
            if rel_path:
                return {"type": "diffuse", "texture": {"type": "image", "path": rel_path}}
        base_color = list(base_color_input.default_value)[:3]
        return {"type": "diffuse", "albedo": color_to_list(base_color)}

    # Try Diffuse BSDF
    node = get_diffuse_bsdf_node(material)
    if node:
        color_input = node.inputs['Color']
        tex_node = get_image_texture_node(color_input)
        if tex_node:
            rel_path = _export_texture_image(tex_node.image, output_dir)
            return {"type": "diffuse", "texture": {"type": "image", "path": rel_path}}
        if obj and color_input.is_linked and _has_complex_shader_tree(color_input):
            rel_path = _bake_material_to_texture(obj, material, output_dir)
            if rel_path:
                return {"type": "diffuse", "texture": {"type": "image", "path": rel_path}}
        color = color_input.default_value
        return {"type": "diffuse", "albedo": color_to_list(color)}

    # Try Emission shader (treat as diffuse with the emission color/texture)
    for nd in material.node_tree.nodes:
        if nd.type == 'EMISSION':
            color_input = nd.inputs['Color']
            tex_node = get_image_texture_node(color_input)
            if tex_node:
                rel_path = _export_texture_image(tex_node.image, output_dir)
                return {"type": "diffuse", "texture": {"type": "image", "path": rel_path}}
            if obj and color_input.is_linked and _has_complex_shader_tree(color_input):
                rel_path = _bake_material_to_texture(obj, material, output_dir)
                if rel_path:
                    return {"type": "diffuse", "texture": {"type": "image", "path": rel_path}}
            color = list(color_input.default_value)[:3]
            return {"type": "diffuse", "albedo": color_to_list(color)}

    # Fallback
    if hasattr(material, 'diffuse_color'):
        return {"type": "diffuse", "albedo": color_to_list(material.diffuse_color)}

    return {"type": "diffuse"}


# ============================================================
# Export functions
# ============================================================

def export_camera(scene, width, height):
    """Export the active camera to Caramel camera dict."""
    cam_obj = scene.camera
    if not cam_obj:
        print("WARNING: No active camera found, using defaults.")
        return {
            "type": "pinhole",
            "pos": [0, 0, 5],
            "dir": [0, 0, -1],
            "up": [0, 1, 0],
            "width": width,
            "height": height,
            "fov": 45
        }

    cam_data = cam_obj.data
    mat = cam_obj.matrix_world

    # Camera position
    pos = mat.translation

    # Blender camera looks down -Z local axis
    direction = mat @ Vector((0, 0, -1)) - pos
    direction.normalize()

    # Up vector
    up = mat @ Vector((0, 1, 0)) - pos
    up.normalize()

    # Field of view — Caramel uses horizontal FOV (fov_x) in degrees
    if cam_data.type == 'PERSP':
        # cam_data.angle is the FOV for the sensor_fit direction
        # For AUTO, Blender uses the larger sensor dimension
        aspect = width / height
        if cam_data.sensor_fit == 'HORIZONTAL':
            fov = math.degrees(cam_data.angle)
        elif cam_data.sensor_fit == 'VERTICAL':
            fov_v = cam_data.angle
            fov = math.degrees(2 * math.atan(math.tan(fov_v / 2) * aspect))
        else:  # AUTO
            if width >= height:
                # cam.angle corresponds to horizontal
                fov = math.degrees(cam_data.angle)
            else:
                # cam.angle corresponds to vertical (height > width)
                fov_v = cam_data.angle
                fov = math.degrees(2 * math.atan(math.tan(fov_v / 2) * aspect))
    else:
        fov = 45.0  # Fallback for orthographic

    return {
        "type": "pinhole",
        "pos": blender_to_caramel_vec(pos),
        "dir": blender_to_caramel_vec(direction),
        "up": blender_to_caramel_vec(up),
        "width": width,
        "height": height,
        "fov": round(fov, 4)
    }


def _export_single_obj(obj, obj_path):
    """Export a single Blender object to an OBJ file."""
    orig_selection = [o for o in bpy.context.selected_objects]
    orig_active = bpy.context.view_layer.objects.active

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if bpy.app.version >= (4, 0, 0):
        bpy.ops.wm.obj_export(
            filepath=obj_path,
            export_selected_objects=True,
            export_materials=False,
            export_triangulated_mesh=True,
            export_normals=True,
            export_uv=True,
            forward_axis='NEGATIVE_Z',
            up_axis='Y',
            apply_modifiers=True,
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=obj_path,
            use_selection=True,
            use_materials=False,
            use_triangles=True,
            use_normals=True,
            use_uvs=True,
            axis_forward='-Z',
            axis_up='Y',
        )

    bpy.ops.object.select_all(action='DESELECT')
    for o in orig_selection:
        o.select_set(True)
    bpy.context.view_layer.objects.active = orig_active


def _split_obj_by_objects(obj_path):
    """
    Split an OBJ file with multiple 'o' headers into separate OBJ files.
    Returns a list of relative paths to the split files.
    """
    mesh_dir = os.path.dirname(obj_path)
    base_name = os.path.splitext(os.path.basename(obj_path))[0]

    # Parse the OBJ file
    with open(obj_path, 'r') as f:
        lines = f.readlines()

    # Collect global vertex/normal/texcoord data and per-object face blocks
    global_header = []  # comment lines before first 'o'
    vertices = []       # (line_index, "v ...")
    normals = []        # (line_index, "vn ...")
    texcoords = []      # (line_index, "vt ...")
    objects = []        # list of {"name": str, "faces": [str]}

    current_obj = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("o "):
            current_obj = {"name": stripped[2:].strip(), "faces": []}
            objects.append(current_obj)
        elif stripped.startswith("v ") and not stripped.startswith("vn") and not stripped.startswith("vt"):
            vertices.append(line)
        elif stripped.startswith("vn "):
            normals.append(line)
        elif stripped.startswith("vt "):
            texcoords.append(line)
        elif stripped.startswith("f ") and current_obj is not None:
            current_obj["faces"].append(line)
        elif stripped.startswith("#"):
            global_header.append(line)

    if len(objects) <= 1:
        # No splitting needed
        return None

    # Remove the original file
    os.remove(obj_path)

    split_paths = []
    for i, obj_block in enumerate(objects):
        # Collect which vertex/normal/texcoord indices are used
        used_v = set()
        used_vn = set()
        used_vt = set()

        for face_line in obj_block["faces"]:
            parts = face_line.strip().split()[1:]  # skip "f"
            for p in parts:
                indices = p.split("/")
                if len(indices) >= 1 and indices[0]:
                    used_v.add(int(indices[0]))
                if len(indices) >= 2 and indices[1]:
                    used_vt.add(int(indices[1]))
                if len(indices) >= 3 and indices[2]:
                    used_vn.add(int(indices[2]))

        # Build remapping (old 1-based index -> new 1-based index)
        sorted_v = sorted(used_v)
        sorted_vn = sorted(used_vn)
        sorted_vt = sorted(used_vt)
        v_remap = {old: new for new, old in enumerate(sorted_v, 1)}
        vn_remap = {old: new for new, old in enumerate(sorted_vn, 1)}
        vt_remap = {old: new for new, old in enumerate(sorted_vt, 1)}

        part_name = f"{base_name}_{i:04d}"
        part_path = os.path.join(mesh_dir, part_name + ".obj")

        with open(part_path, 'w') as f:
            for h in global_header:
                f.write(h)
            f.write(f"o {part_name}\n")

            for idx in sorted_v:
                f.write(vertices[idx - 1])
            for idx in sorted_vn:
                f.write(normals[idx - 1])
            for idx in sorted_vt:
                f.write(texcoords[idx - 1])

            for face_line in obj_block["faces"]:
                parts = face_line.strip().split()
                new_parts = ["f"]
                for p in parts[1:]:
                    indices = p.split("/")
                    new_indices = []
                    if len(indices) >= 1 and indices[0]:
                        new_indices.append(str(v_remap[int(indices[0])]))
                    else:
                        new_indices.append("")
                    if len(indices) >= 2 and indices[1]:
                        new_indices.append(str(vt_remap[int(indices[1])]))
                    elif len(indices) >= 2:
                        new_indices.append("")
                    if len(indices) >= 3 and indices[2]:
                        new_indices.append(str(vn_remap[int(indices[2])]))
                    new_parts.append("/".join(new_indices))
                f.write(" ".join(new_parts) + "\n")

        split_paths.append(os.path.join("meshes", part_name + ".obj"))

    return split_paths


def _get_material_id(material):
    """Return a stable ID string for a Blender material."""
    if not material:
        return "default"
    return bpy.path.clean_name(material.name)


def _get_face_materials(obj):
    """
    Get per-face material assignment from the evaluated mesh.
    Returns a dict mapping material index -> (material_or_None, [face_indices]).
    If all faces share one material, returns a single entry.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.data

    mat_faces = {}
    for i, poly in enumerate(eval_mesh.polygons):
        idx = poly.material_index
        if idx not in mat_faces:
            mat = None
            if idx < len(eval_obj.material_slots) and eval_obj.material_slots[idx].material:
                mat = eval_obj.material_slots[idx].material
            mat_faces[idx] = (mat, [])
        mat_faces[idx][1].append(i)

    return mat_faces


def _export_obj_by_material(obj, output_dir):
    """
    Export a mesh object as separate OBJ files per material.
    Returns list of (rel_path, material) tuples.
    """
    import bmesh

    mesh_dir = os.path.join(output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    name = bpy.path.clean_name(obj.name)
    mat_faces = _get_face_materials(obj)

    # If only one material (or none), no need to split
    if len(mat_faces) <= 1:
        return None

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)

    results = []
    for mat_idx, (material, face_indices) in mat_faces.items():
        mat_id = _get_material_id(material)
        part_name = f"{name}_{mat_id}"
        obj_path = os.path.join(mesh_dir, part_name + ".obj")
        rel_path = os.path.join("meshes", part_name + ".obj")

        face_set = set(face_indices)

        bm = bmesh.new()
        bm.from_object(eval_obj, depsgraph)
        bm.faces.ensure_lookup_table()

        # Delete faces not in this material group
        faces_to_delete = [f for f in bm.faces if f.index not in face_set]
        bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')

        # Transform vertices to world space (bmesh gives local coords)
        bm.transform(obj.matrix_world)

        # Create a temporary mesh and object for export
        tmp_mesh = bpy.data.meshes.new(part_name)
        bm.to_mesh(tmp_mesh)
        bm.free()

        tmp_obj = bpy.data.objects.new(part_name, tmp_mesh)
        bpy.context.collection.objects.link(tmp_obj)

        _export_single_obj(tmp_obj, obj_path)

        # Clean up temporary objects
        bpy.data.objects.remove(tmp_obj, do_unlink=True)
        bpy.data.meshes.remove(tmp_mesh)

        results.append((rel_path, material))

    return results


def _has_particle_systems(obj):
    """Check if an object has particle systems that instance other objects."""
    return len(obj.particle_systems) > 0


def _export_particle_instances(obj, output_dir, split_objects=False):
    """
    Export particle system instances as realized meshes.
    For each particle system on obj, this converts particle instances into
    actual mesh data and exports them as OBJ files.
    Returns list of Caramel shape dicts, or None if no particle systems.
    """
    if not _has_particle_systems(obj):
        return None

    mesh_dir = os.path.join(output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # Enable viewport display for particle modifiers so depsgraph evaluates them
    orig_viewport_states = {}
    for mod in obj.modifiers:
        if mod.type == 'PARTICLE_SYSTEM':
            orig_viewport_states[mod.name] = mod.show_viewport
            mod.show_viewport = True

    # Force depsgraph update
    bpy.context.scene.frame_set(bpy.context.scene.frame_current)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    shapes = []

    # Collect instance objects and their materials from particle settings
    instance_materials = {}  # instance obj name -> material
    for ps in obj.particle_systems:
        settings = ps.settings
        if settings.render_type == 'OBJECT' and settings.instance_object:
            inst_obj = settings.instance_object
            mat = inst_obj.active_material
            instance_materials[inst_obj.name] = mat
        elif settings.render_type == 'COLLECTION' and settings.instance_collection:
            for inst_obj in settings.instance_collection.objects:
                mat = inst_obj.active_material
                instance_materials[inst_obj.name] = mat

    # Use depsgraph to iterate over all object instances
    # Collect per-source-object instance transforms
    source_instances = {}  # source obj name -> list of Matrix4x4
    for dup in depsgraph.object_instances:
        if dup.parent and dup.parent.original == obj and dup.is_instance:
            src_name = dup.object.name
            if src_name not in source_instances:
                source_instances[src_name] = []
            source_instances[src_name].append(dup.matrix_world.copy())

    if not source_instances:
        return None

    print(f"  Particle instances on '{obj.name}':")
    for src_name, matrices in source_instances.items():
        print(f"    {src_name}: {len(matrices)} instances")

    # For each source object, create a combined mesh with all instances
    for src_name, matrices in source_instances.items():
        import bmesh

        # Get the source object's evaluated mesh
        src_obj = bpy.data.objects.get(src_name)
        if not src_obj:
            continue

        eval_src = src_obj.evaluated_get(depsgraph)

        # Combine all instances into a single mesh
        combined_bm = bmesh.new()
        for mat in matrices:
            bm_inst = bmesh.new()
            bm_inst.from_object(eval_src, depsgraph)
            # Apply instance world transform (OBJ exporter handles coord conversion)
            bmesh.ops.transform(bm_inst, matrix=mat, verts=bm_inst.verts)
            # Merge into combined
            temp_mesh = bpy.data.meshes.new(f"_temp_{src_name}")
            bm_inst.to_mesh(temp_mesh)
            bm_inst.free()
            combined_bm.from_mesh(temp_mesh)
            bpy.data.meshes.remove(temp_mesh)

        # Create a temporary mesh and object for export
        part_name = bpy.path.clean_name(f"{obj.name}_particles_{src_name}")
        combined_mesh = bpy.data.meshes.new(part_name)
        combined_bm.to_mesh(combined_mesh)
        combined_bm.free()

        tmp_obj = bpy.data.objects.new(part_name, combined_mesh)
        bpy.context.collection.objects.link(tmp_obj)

        obj_path = os.path.join(mesh_dir, part_name + ".obj")
        rel_path = os.path.join("meshes", part_name + ".obj")

        _export_single_obj(tmp_obj, obj_path)

        # Cleanup
        bpy.data.objects.remove(tmp_obj, do_unlink=True)
        bpy.data.meshes.remove(combined_mesh)

        # Determine material
        mat = instance_materials.get(src_name)
        bsdf_id = _get_material_id(mat)

        shape = {"type": "obj", "path": rel_path, "bsdf": bsdf_id}
        shapes.append(shape)

        print(f"    Exported combined particle mesh: {part_name}.obj")

    # Restore viewport states
    for mod in obj.modifiers:
        if mod.type == 'PARTICLE_SYSTEM' and mod.name in orig_viewport_states:
            mod.show_viewport = orig_viewport_states[mod.name]

    return shapes if shapes else None


def export_mesh(obj, output_dir, split_objects=False):
    """Export a mesh object to OBJ and return Caramel shape dict(s).
    Note: OBJ exporter applies matrix_world + axis conversion, so vertices
    are already in Caramel world space. No to_world transform needed."""
    mesh_dir = os.path.join(output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    name = bpy.path.clean_name(obj.name)

    # Export particle instances if present
    particle_shapes = _export_particle_instances(obj, output_dir, split_objects)

    # Try splitting by material first
    mat_split = _export_obj_by_material(obj, output_dir)
    if mat_split:
        print(f"  Split '{obj.name}' by material into {len(mat_split)} parts")
        shapes = []
        for rel_path, material in mat_split:
            bsdf_id = _get_material_id(material)
            shape = {"type": "obj", "path": rel_path, "bsdf": bsdf_id}
            shapes.append(shape)
        if particle_shapes:
            shapes.extend(particle_shapes)
        return shapes

    # Single material — export as one OBJ
    obj_path = os.path.join(mesh_dir, name + ".obj")
    rel_path = os.path.join("meshes", name + ".obj")
    _export_single_obj(obj, obj_path)

    material = obj.active_material if obj.active_material else None
    bsdf_id = _get_material_id(material)

    if split_objects:
        split_paths = _split_obj_by_objects(obj_path)
        if split_paths:
            print(f"  Split '{obj.name}' into {len(split_paths)} objects")
            shapes = []
            for sp in split_paths:
                shape = {"type": "obj", "path": sp, "bsdf": bsdf_id}
                shapes.append(shape)
            if particle_shapes:
                shapes.extend(particle_shapes)
            return shapes

    shape = {"type": "obj", "path": rel_path, "bsdf": bsdf_id}
    shapes = [shape]
    if particle_shapes:
        shapes.extend(particle_shapes)
    return shapes


def export_lights(scene):
    """Export Blender lights to Caramel light array."""
    lights = []

    for obj in scene.objects:
        if obj.type != 'LIGHT':
            continue

        light_data = obj.data

        if light_data.type == 'POINT':
            energy = light_data.energy
            color = light_data.color
            radiance = [round(float(color[0] * energy), 4),
                        round(float(color[1] * energy), 4),
                        round(float(color[2] * energy), 4)]
            lights.append({
                "type": "point",
                "pos": blender_to_caramel_vec(obj.location),
                "radiance": radiance
            })

        elif light_data.type == 'SUN':
            # Approximate sun as constant environment light
            energy = light_data.energy
            color = light_data.color
            radiance = color_to_list(color)
            lights.append({
                "type": "constant_env",
                "radiance": radiance,
                "scale": round(float(energy), 4)
            })

        elif light_data.type == 'AREA':
            # Area lights in Blender are converted to mesh emitters in Caramel
            # Handled separately in export_area_light_as_shape
            pass

    # Check for environment/world lighting
    world = scene.world
    if world and world.use_nodes:
        env_light = _export_world_light(world, scene)
        if env_light:
            lights.append(env_light)

    return lights


def _export_world_light(world, scene):
    """Convert Blender world environment to Caramel environment light."""
    if not world.use_nodes:
        return None

    for node in world.node_tree.nodes:
        if node.type == 'BACKGROUND':
            strength = node.inputs['Strength'].default_value
            color = list(node.inputs['Color'].default_value)[:3]

            # Check if the color input is connected to an environment texture
            color_socket = node.inputs['Color']
            if color_socket.is_linked:
                linked_node = color_socket.links[0].from_node
                if linked_node.type == 'TEX_ENVIRONMENT' and linked_node.image:
                    # Export the HDR image
                    image = linked_node.image
                    # Try to find the original file path
                    src_path = bpy.path.abspath(image.filepath)
                    if os.path.exists(src_path):
                        import shutil
                        # Determine output scene dir from first arg or OUTPUT_DIR
                        args = parse_args()
                        out_dir = args.output_dir if args else OUTPUT_DIR
                        dst_name = os.path.basename(src_path)
                        dst_path = os.path.join(out_dir, dst_name)
                        if src_path != dst_path:
                            shutil.copy2(src_path, dst_path)
                        return {
                            "type": "image_env",
                            "path": dst_name,
                            "scale": round(float(strength), 4)
                        }

            # Constant environment
            if strength > 0:
                return {
                    "type": "constant_env",
                    "radiance": color_to_list(color),
                    "scale": round(float(strength), 4)
                }

    return None


def export_area_light_shapes(scene, output_dir):
    """
    Convert Blender AREA lights into Caramel mesh shapes with arealight property.
    Creates a simple quad mesh for each area light.
    """
    shapes = []

    for obj in scene.objects:
        if obj.type != 'LIGHT':
            continue
        light_data = obj.data
        if light_data.type != 'AREA':
            continue

        energy = light_data.energy
        color = light_data.color
        radiance = [round(float(color[0] * energy), 4),
                    round(float(color[1] * energy), 4),
                    round(float(color[2] * energy), 4)]

        # Create a temporary plane mesh for the area light
        size_x = light_data.size
        size_y = light_data.size_y if light_data.shape == 'RECTANGLE' else light_data.size
        half_x = size_x / 2
        half_y = size_y / 2

        mesh_dir = os.path.join(output_dir, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)

        name = bpy.path.clean_name(obj.name)
        obj_path = os.path.join(mesh_dir, name + ".obj")
        rel_path = os.path.join("meshes", name + ".obj")

        # Write a simple quad OBJ
        # Blender area lights emit in local -Z, so face normal should be (0,0,-1)
        with open(obj_path, 'w') as f:
            f.write(f"# Area light: {obj.name}\n")
            f.write(f"v {-half_x} {-half_y} 0\n")
            f.write(f"v { half_x} {-half_y} 0\n")
            f.write(f"v { half_x} { half_y} 0\n")
            f.write(f"v {-half_x} { half_y} 0\n")
            f.write("vn 0 0 -1\n")
            f.write("f 1//1 3//1 2//1\n")
            f.write("f 1//1 4//1 3//1\n")

        shape = {
            "type": "obj",
            "path": rel_path,
            "bsdf": {"type": "diffuse"},
            "arealight": {"radiance": radiance},
        }

        to_world = decompose_transform(obj)
        if to_world:
            shape["to_world"] = to_world

        shapes.append(shape)

    return shapes


def check_emissive_material(obj):
    """Check if an object has an emissive material, return radiance if so."""
    mat = obj.active_material
    if not mat or not mat.use_nodes:
        return None

    for node in mat.node_tree.nodes:
        if node.type == 'EMISSION':
            strength = node.inputs['Strength'].default_value
            color = list(node.inputs['Color'].default_value)[:3]
            return [round(float(c * strength), 4) for c in color]

        if node.type == 'BSDF_PRINCIPLED':
            # Check emission on Principled BSDF
            emission_color = list(node.inputs['Emission Color'].default_value)[:3] \
                if 'Emission Color' in node.inputs else [0, 0, 0]

            emission_strength = 1.0
            if 'Emission Strength' in node.inputs:
                emission_strength = node.inputs['Emission Strength'].default_value
            elif 'Emission' in node.inputs:
                # Older Blender versions
                emission_color = list(node.inputs['Emission'].default_value)[:3]

            total = sum(emission_color)
            if total > 0 and emission_strength > 0:
                return [round(float(c * emission_strength), 4) for c in emission_color]

    return None


def _get_particle_instance_objects(scene):
    """Collect objects that are used only as particle system instances.
    These should not be exported as standalone shapes."""
    instance_objs = set()
    for obj in scene.objects:
        for ps in obj.particle_systems:
            settings = ps.settings
            if settings.render_type == 'OBJECT' and settings.instance_object:
                instance_objs.add(settings.instance_object.name)
            elif settings.render_type == 'COLLECTION' and settings.instance_collection:
                for inst_obj in settings.instance_collection.objects:
                    instance_objs.add(inst_obj.name)
    return instance_objs


# ============================================================
# Main export
# ============================================================

def export_scene():
    """Main export function."""
    args = parse_args()
    if args:
        output_dir = args.output_dir
        spp = args.spp
        depth_max = args.depth_max
        depth_rr = args.depth_rr
        width = args.width
        height = args.height
        split_objects = not args.no_split_objects
    else:
        output_dir = OUTPUT_DIR
        spp = DEFAULT_SPP
        depth_max = DEFAULT_DEPTH_MAX
        depth_rr = DEFAULT_DEPTH_RR
        width = None
        height = None
        split_objects = True

    os.makedirs(output_dir, exist_ok=True)

    scene = bpy.context.scene

    # Resolution: use args or Blender render settings
    if width is None:
        width = scene.render.resolution_x
    if height is None:
        height = scene.render.resolution_y

    # Build scene dict
    caramel_scene = {}

    # Integrator
    caramel_scene["integrator"] = {
        "type": "path",
        "depth_rr": depth_rr,
        "depth_max": depth_max,
        "spp": spp
    }

    # Camera
    caramel_scene["camera"] = export_camera(scene, width, height)

    # Lights
    lights = export_lights(scene)
    if lights:
        caramel_scene["light"] = lights

    # Collect unique BSDFs from all mesh objects (including evaluated materials)
    bsdf_map = {}  # id -> bsdf dict
    for obj in scene.objects:
        if obj.type != 'MESH' or not obj.visible_get():
            continue
        # Check all materials from evaluated mesh (geometry nodes etc.)
        mat_faces = _get_face_materials(obj)
        for mat_idx, (material, _) in mat_faces.items():
            bsdf_id = _get_material_id(material)
            if bsdf_id not in bsdf_map:
                bsdf = convert_material(material, output_dir, obj)
                bsdf["id"] = bsdf_id
                bsdf_map[bsdf_id] = bsdf

    if bsdf_map:
        caramel_scene["bsdfs"] = list(bsdf_map.values())

    # Shapes
    shapes = []

    # Collect objects used only as particle instances (skip standalone export)
    particle_instance_objs = _get_particle_instance_objects(scene)

    # Export mesh objects
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        if not obj.visible_get():
            continue
        if obj.name in particle_instance_objs:
            print(f"Skipping particle instance source: {obj.name}")
            continue

        print(f"Exporting mesh: {obj.name}")
        exported_shapes = export_mesh(obj, output_dir, split_objects)

        # Check for emissive material → arealight
        emissive_radiance = check_emissive_material(obj)
        if emissive_radiance:
            for s in exported_shapes:
                s["arealight"] = {"radiance": emissive_radiance}

        shapes.extend(exported_shapes)

    # Export area lights as mesh shapes
    area_light_shapes = export_area_light_shapes(scene, output_dir)
    shapes.extend(area_light_shapes)

    caramel_scene["shape"] = shapes

    # Write JSON
    output_path = os.path.join(output_dir, "scene.json")
    with open(output_path, 'w') as f:
        json.dump(caramel_scene, f, indent=4)

    print(f"\n{'='*50}")
    print(f"Caramel scene exported to: {output_path}")
    print(f"  Meshes: {len(shapes)}")
    print(f"  Lights: {len(lights)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  SPP: {spp}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    export_scene()
