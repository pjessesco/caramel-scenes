#!/usr/bin/env python3

import os
import sys
import subprocess
import re

# ==========================================
# Configuration
# ==========================================
# Keyword and value to overwrite
TARGET_KEY = "spp"
TARGET_VALUE = 1

# Output filename to rename
OUTPUT_FILENAME = "gt.exr"
# ==========================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 render_gt.py <path_to_caramel_executable>")
        sys.exit(1)

    caramel_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(caramel_path):
        print(f"Error: Executable not found at {caramel_path}")
        sys.exit(1)

    # Directory where the script is located (caramel-scenes)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Collect all scenes to render (recursive)
    scenes = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames.sort()
        json_files = sorted([f for f in filenames if f.endswith(".json")])
        if json_files:
            rel_dir = os.path.relpath(dirpath, base_dir)
            scenes.append((rel_dir, dirpath, json_files))

    # Print summary before rendering
    total = sum(len(jf) for _, _, jf in scenes)
    print(f"Found {total} scene(s) in {len(scenes)} directory(ies):")
    for rel_dir, _, json_files in scenes:
        for jf in json_files:
            print(f"  - {rel_dir}/{jf}")
    print()

    # Track rendered results
    results = []

    # Iterate over all folders in caramel-scenes
    for rel_dir, scene_dir, json_files in scenes:
        print(f"Processing directory: {rel_dir}")

        for json_file in json_files:
            scene_json_path = os.path.join(scene_dir, json_file)
            print(f"  Processing {json_file}...")

            # 1. Read and modify JSON file (text-based to preserve formatting)
            try:
                with open(scene_json_path, 'r') as f:
                    content = f.read()

                # Replace spp value in-place using regex
                new_content, count = re.subn(
                    r'("' + TARGET_KEY + r'"\s*:\s*)\d+',
                    lambda m: m.group(1) + str(TARGET_VALUE),
                    content
                )

                if count > 0:
                    print(f"    Overwriting {TARGET_KEY} to {TARGET_VALUE} ({count} occurrence(s))")
                    with open(scene_json_path, 'w') as f:
                        f.write(new_content)
                else:
                    print(f"    Warning: '{TARGET_KEY}' key not found in {json_file}. Skipping modification.")

            except Exception as e:
                print(f"    Error modifying JSON: {e}")
                continue

            # 2. Execute rendering
            # Execute using absolute path
            try:
                print(f"    Running renderer...", flush=True)
                # Remove cwd argument and pass absolute path
                subprocess.run([caramel_path, scene_json_path], check=True, stdout=sys.stdout, stderr=sys.stderr)
            except subprocess.CalledProcessError as e:
                print(f"    Rendering failed for {json_file}: {e}")
                continue

            # 3. Rename output file
            # caramel generates .exr using the input filename (stem)
            stem = os.path.splitext(json_file)[0]
            default_output = os.path.join(scene_dir, stem + ".exr")
            target_output = os.path.join(scene_dir, OUTPUT_FILENAME)

            if os.path.exists(default_output):
                # Add json filename as postfix if multiple json files in directory
                if len(json_files) > 1:
                    name, ext = os.path.splitext(OUTPUT_FILENAME)
                    target_output = os.path.join(scene_dir, f"{name}_{stem}{ext}")
                os.rename(default_output, target_output)
                print(f"    Renamed output to {os.path.basename(target_output)}")
                results.append((scene_json_path, target_output))
            else:
                print(f"    Warning: Output file {default_output} not found.")

    # Print render summary
    if results:
        print()
        print(f"Render complete ({len(results)} file(s)):")
        for src, dst in results:
            print(f"  {src} -> {dst}")

if __name__ == "__main__":
    main()
