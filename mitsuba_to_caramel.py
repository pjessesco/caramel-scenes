#!/usr/bin/env python3
"""
Convert Mitsuba 3 scene XML to Caramel scene JSON.

Usage:
    python scripts/mitsuba_to_caramel.py caramel-scenes/house/scene_v3.xml [-o output.json]

Supported Mitsuba features:
  - perspective sensor → camera (pos/dir/up or matrix)
  - diffuse BSDF (rgb reflectance, bitmap texture)
  - thindielectric / dielectric BSDF → dielectric
  - envmap emitter → image_env light
  - obj shapes with to_world matrix transforms
  - BSDF refs (id/ref pattern)
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_defaults(root):
    """Parse <default> elements into a dict."""
    defaults = {}
    for d in root.findall("default"):
        defaults[d.get("name")] = d.get("value")
    return defaults


def resolve(value, defaults):
    """Resolve $variable references using defaults dict."""
    if isinstance(value, str) and value.startswith("$"):
        return defaults.get(value[1:], value)
    return value


def parse_rgb(value_str):
    """Parse 'r, g, b' string to [r, g, b] list."""
    return [float(x.strip()) for x in value_str.split(",")]


def parse_matrix_values(value_str):
    """Parse '16 space-separated floats' to 4x4 row-major list."""
    return [float(x) for x in value_str.split()]


def is_identity_matrix(mat_values):
    """Check if 16 float values represent an identity matrix."""
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    return all(abs(a - b) < 1e-9 for a, b in zip(mat_values, identity))


def parse_bsdf_element(bsdf_elem):
    """Parse a single <bsdf> element into a Caramel BSDF dict."""
    bsdf_type = bsdf_elem.get("type")

    if bsdf_type == "twosided":
        inner_bsdfs = bsdf_elem.findall("bsdf")
        if len(inner_bsdfs) >= 1:
            front = parse_bsdf_element(inner_bsdfs[0])
            result = {"type": "twosided", "bsdf": front}
            if len(inner_bsdfs) >= 2:
                result["back_bsdf"] = parse_bsdf_element(inner_bsdfs[1])
            return result

    if bsdf_type == "diffuse":
        result = {"type": "diffuse"}
        rgb = bsdf_elem.find("rgb[@name='reflectance']")
        tex = bsdf_elem.find("texture[@name='reflectance']")
        if tex is not None:
            filename = tex.find("string[@name='filename']")
            if filename is not None:
                result["texture"] = {
                    "type": "image",
                    "path": filename.get("value"),
                }
        elif rgb is not None:
            result["albedo"] = parse_rgb(rgb.get("value"))
        return result

    if bsdf_type in ("thindielectric", "dielectric"):
        result = {"type": "dielectric"}
        int_ior = bsdf_elem.find("float[@name='int_ior']")
        ext_ior = bsdf_elem.find("float[@name='ext_ior']")
        result["in_ior"] = float(int_ior.get("value")) if int_ior is not None else 1.5
        result["ex_ior"] = float(ext_ior.get("value")) if ext_ior is not None else 1.0
        return result

    if bsdf_type == "conductor":
        result = {"type": "conductor"}
        material = bsdf_elem.find("string[@name='material']")
        result["material"] = material.get("value") if material is not None else "Cu"
        result["ex_ior"] = 1.0
        return result

    if bsdf_type == "roughconductor":
        result = {"type": "conductor"}
        material = bsdf_elem.find("string[@name='material']")
        result["material"] = material.get("value") if material is not None else "Cu"
        result["ex_ior"] = 1.0
        return result

    if bsdf_type in ("plastic", "roughplastic"):
        result = {"type": "microfacet"}
        alpha = bsdf_elem.find("float[@name='alpha']")
        result["alpha"] = float(alpha.get("value")) if alpha is not None else 0.1
        int_ior = bsdf_elem.find("float[@name='int_ior']")
        ext_ior = bsdf_elem.find("float[@name='ext_ior']")
        result["in_ior"] = float(int_ior.get("value")) if int_ior is not None else 1.5
        result["ex_ior"] = float(ext_ior.get("value")) if ext_ior is not None else 1.0
        rgb = bsdf_elem.find("rgb[@name='diffuse_reflectance']")
        if rgb is not None:
            result["kd"] = parse_rgb(rgb.get("value"))
        else:
            result["kd"] = [0.5, 0.5, 0.5]
        return result

    if bsdf_type == "mirror":
        return {"type": "mirror"}

    print(f"  [WARN] Unsupported BSDF type '{bsdf_type}', falling back to diffuse",
          file=sys.stderr)
    return {"type": "diffuse"}


def parse_transform(transform_elem):
    """Parse <transform> element into to_world for Caramel, or None if identity."""
    if transform_elem is None:
        return None

    matrix_elem = transform_elem.find("matrix")
    if matrix_elem is not None:
        mat_values = parse_matrix_values(matrix_elem.get("value"))
        if is_identity_matrix(mat_values):
            return None
        return mat_values

    transforms = []
    for child in transform_elem:
        if child.tag == "translate":
            x = float(child.get("x", "0"))
            y = float(child.get("y", "0"))
            z = float(child.get("z", "0"))
            if abs(x) > 1e-9 or abs(y) > 1e-9 or abs(z) > 1e-9:
                transforms.append({"type": "translate", "value": [x, y, z]})
        elif child.tag == "rotate":
            angle = float(child.get("angle", "0"))
            x = float(child.get("x", "0"))
            y = float(child.get("y", "0"))
            z = float(child.get("z", "0"))
            if abs(x) > 0.5:
                transforms.append({"type": "rotate_x", "degree": angle})
            elif abs(y) > 0.5:
                transforms.append({"type": "rotate_y", "degree": angle})
            elif abs(z) > 0.5:
                transforms.append({"type": "rotate_z", "degree": angle})
        elif child.tag == "scale":
            x = float(child.get("x", child.get("value", "1")))
            y = float(child.get("y", child.get("value", "1")))
            z = float(child.get("z", child.get("value", "1")))
            transforms.append({"type": "scale", "value": [x, y, z]})

    return transforms if transforms else None


def convert(xml_path):
    xml_path = Path(xml_path)
    output_path = xml_path.with_name("scene.json")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    defaults = parse_defaults(root)

    scene = {}

    # --- Integrator ---
    integrator_elem = root.find("integrator")
    integrator_type = resolve(integrator_elem.get("type"), defaults) if integrator_elem is not None else "path"
    max_depth_elem = integrator_elem.find("integer[@name='max_depth']") if integrator_elem is not None else None
    max_depth = int(resolve(max_depth_elem.get("value"), defaults)) if max_depth_elem is not None else 5
    spp = int(resolve(defaults.get("spp", "64"), defaults))

    scene["integrator"] = {
        "type": integrator_type if integrator_type in ("path",) else "path",
        "depth_rr": 0,
        "depth_max": min(max_depth, 30),
        "spp": spp,
    }

    # --- Camera (sensor) ---
    sensor = root.find("sensor")
    if sensor is not None:
        fov_elem = sensor.find("float[@name='fov']")
        fov = float(fov_elem.get("value")) if fov_elem is not None else 45.0

        film = sensor.find("film")
        width = int(resolve(defaults.get("resx", "1280"), defaults))
        height = int(resolve(defaults.get("resy", "720"), defaults))
        if film is not None:
            w_elem = film.find("integer[@name='width']")
            h_elem = film.find("integer[@name='height']")
            if w_elem is not None:
                width = int(resolve(w_elem.get("value"), defaults))
            if h_elem is not None:
                height = int(resolve(h_elem.get("value"), defaults))

        transform = sensor.find("transform[@name='to_world']")
        if transform is not None:
            matrix_elem = transform.find("matrix")
            lookat = transform.find("lookat")
            if matrix_elem is not None:
                # Pass cam-to-world matrix directly (both Mitsuba and Caramel use row-major)
                mat = parse_matrix_values(matrix_elem.get("value"))
                scene["camera"] = {
                    "type": "perspective",
                    "matrix": mat,
                    "width": width,
                    "height": height,
                    "fov": fov,
                }
            elif lookat is not None:
                origin = [float(x) for x in lookat.get("origin").split(",")]
                target = [float(x) for x in lookat.get("target").split(",")]
                up = [float(x) for x in lookat.get("up", "0,1,0").split(",")]
                dir_ = [t - o for t, o in zip(target, origin)]
                length = math.sqrt(sum(x * x for x in dir_))
                dir_ = [x / length for x in dir_]
                scene["camera"] = {
                    "type": "perspective",
                    "pos": origin,
                    "dir": [round(x, 6) for x in dir_],
                    "up": up,
                    "width": width,
                    "height": height,
                    "fov": fov,
                }
            else:
                scene["camera"] = {
                    "type": "perspective",
                    "pos": [0, 0, 5],
                    "dir": [0, 0, -1],
                    "up": [0, 1, 0],
                    "width": width,
                    "height": height,
                    "fov": fov,
                }
        else:
            scene["camera"] = {
                "type": "perspective",
                "pos": [0, 0, 5],
                "dir": [0, 0, -1],
                "up": [0, 1, 0],
                "width": width,
                "height": height,
                "fov": fov,
            }

    # --- Collect named BSDFs ---
    bsdf_map = {}
    bsdfs_list = []
    for bsdf_elem in root.findall("bsdf"):
        bsdf_id = bsdf_elem.get("id")
        if bsdf_id:
            parsed = parse_bsdf_element(bsdf_elem)
            parsed["id"] = bsdf_id
            bsdf_map[bsdf_id] = bsdf_id
            bsdfs_list.append(parsed)

    if bsdfs_list:
        scene["bsdfs"] = bsdfs_list

    # --- Shapes ---
    shapes = []
    for shape_elem in root.findall("shape"):
        shape_type = shape_elem.get("type")
        if shape_type != "obj":
            print(f"  [WARN] Skipping unsupported shape type '{shape_type}'",
                  file=sys.stderr)
            continue

        filename_elem = shape_elem.find("string[@name='filename']")
        if filename_elem is None:
            continue
        filename = filename_elem.get("value")

        # Resolve BSDF
        bsdf = None
        ref_elem = shape_elem.find("ref")
        if ref_elem is not None:
            bsdf_id = ref_elem.get("id")
            if bsdf_id in bsdf_map:
                bsdf = bsdf_id
            else:
                print(f"  [WARN] BSDF ref '{bsdf_id}' not found, using default diffuse",
                      file=sys.stderr)
                bsdf = {"type": "diffuse"}
        else:
            inner_bsdf = shape_elem.find("bsdf")
            if inner_bsdf is not None:
                bsdf = parse_bsdf_element(inner_bsdf)
            else:
                bsdf = {"type": "diffuse"}

        shape = {
            "type": "obj",
            "path": filename,
            "bsdf": bsdf,
        }

        # Transform
        transform = shape_elem.find("transform[@name='to_world']")
        to_world = parse_transform(transform)
        if to_world is not None:
            shape["to_world"] = to_world

        # Area light emitter inside shape
        emitter = shape_elem.find("emitter")
        if emitter is not None:
            rad_elem = emitter.find("rgb[@name='radiance']")
            if rad_elem is not None:
                shape["arealight"] = {
                    "radiance": parse_rgb(rad_elem.get("value")),
                }

        shapes.append(shape)

    scene["shape"] = shapes

    # --- Lights (scene-level emitters) ---
    lights = []
    for emitter_elem in root.findall("emitter"):
        emitter_type = emitter_elem.get("type")

        if emitter_type == "envmap":
            filename_elem = emitter_elem.find("string[@name='filename']")
            if filename_elem is not None:
                light = {
                    "type": "image_env",
                    "path": filename_elem.get("value"),
                    "scale": 1,
                }
                transform = emitter_elem.find("transform")
                to_world = parse_transform(transform)
                if to_world is not None:
                    light["to_world"] = to_world
                lights.append(light)

        elif emitter_type == "point":
            pos_elem = emitter_elem.find("point[@name='position']")
            rad_elem = emitter_elem.find("rgb[@name='intensity']")
            if pos_elem is not None and rad_elem is not None:
                lights.append({
                    "type": "point",
                    "pos": [float(pos_elem.get("x", "0")),
                            float(pos_elem.get("y", "0")),
                            float(pos_elem.get("z", "0"))],
                    "radiance": parse_rgb(rad_elem.get("value")),
                })

        elif emitter_type == "constant":
            rad_elem = emitter_elem.find("rgb[@name='radiance']")
            if rad_elem is not None:
                lights.append({
                    "type": "constant_env",
                    "radiance": parse_rgb(rad_elem.get("value")),
                    "scale": 1,
                })

        else:
            print(f"  [WARN] Unsupported emitter type '{emitter_type}'",
                  file=sys.stderr)

    if lights:
        scene["light"] = lights

    # --- Output ---
    json_str = json.dumps(scene, indent=4)
    output_path.write_text(json_str + "\n")
    print(f"Written to {output_path}", file=sys.stderr)

    print(f"\n--- Conversion summary ---", file=sys.stderr)
    print(f"  Shapes: {len(shapes)}", file=sys.stderr)
    print(f"  Named BSDFs: {len(bsdf_map)}", file=sys.stderr)
    print(f"  Lights: {len(lights)}", file=sys.stderr)
    print(f"  SPP: {spp}, Max depth: {scene['integrator']['depth_max']}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Mitsuba 3 scene XML to Caramel scene JSON"
    )
    parser.add_argument("input", help="Path to Mitsuba scene XML file")
    args = parser.parse_args()

    convert(args.input)


if __name__ == "__main__":
    main()
