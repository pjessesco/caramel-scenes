#!/usr/bin/env python3

import os
import sys
import json
import subprocess
import re

# ==========================================
# Configuration
# ==========================================
# Keyword and value to overwrite
TARGET_KEY = "spp"
TARGET_VALUE = 3000

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

    # Iterate over all folders in caramel-scenes
    for item in os.listdir(base_dir):
        scene_dir = os.path.join(base_dir, item)

        # Check if it is a directory
        if os.path.isdir(scene_dir):
            # Find all .json files in the directory
            json_files = [f for f in os.listdir(scene_dir) if f.endswith(".json")]

            if not json_files:
                continue

            print(f"Processing directory: {item}")

            for json_file in json_files:
                scene_json_path = os.path.join(scene_dir, json_file)
                print(f"  Processing {json_file}...")

                # 1. Read and modify JSON file
                try:
                    with open(scene_json_path, 'r') as f:
                        data = json.load(f)

                    if "integrator" in data:
                        print(f"    Overwriting integrator.{TARGET_KEY} to {TARGET_VALUE}")
                        data["integrator"][TARGET_KEY] = TARGET_VALUE

                        # JSON formatting: indent=4, but display number arrays on one line
                        json_str = json.dumps(data, indent=4)

                        # Find arrays containing only numbers and remove line breaks
                        # Example: [ \n 1, \n 2 \n ] -> [ 1, 2 ]
                        json_str = re.sub(r'\[\s*([\d\.\,\-\seE]+?)\s*\]',
                                          lambda m: "[" + re.sub(r'\s+', ' ', m.group(1)).strip() + "]",
                                          json_str)

                        with open(scene_json_path, 'w') as f:
                            f.write(json_str)
                    else:
                        print(f"    Warning: 'integrator' key not found in {json_file}. Skipping modification.")

                except json.JSONDecodeError:
                    print(f"    Error: Failed to parse JSON in {json_file}. Skipping.")
                    continue
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
                    os.rename(default_output, target_output)
                    print(f"    Renamed output to {OUTPUT_FILENAME}")
                else:
                    print(f"    Warning: Output file {default_output} not found.")

if __name__ == "__main__":
    main()
