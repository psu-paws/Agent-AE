import os
import json

log_dir = "../../logs/gaia-validation-text-103/openai_gpt-4o-mini_mirothinker_v1.5_keep20_max200/run_1/"

def find_tool_paths(data, target, path="root"):
    paths = []
    if isinstance(data, dict):
        for k, v in data.items():
            paths.extend(find_tool_paths(v, target, f"{path}['{k}']"))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            paths.extend(find_tool_paths(v, target, f"{path}[{i}]"))
    elif isinstance(data, str) and target in data:
        # We found the target! Truncate the string so the terminal doesn't flood
        snippet = data.replace('\n', ' ')[:150]
        paths.append(f"{path} \n    -> Contains: '{snippet}...'")
    return paths

print("Hunting for the tool schema...\n")

for root, dirs, files in os.walk(log_dir):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Quick check if this file contains a search
                    if "google_search" in json.dumps(data):
                        print(f"--- Found 'google_search' in {file} ---")
                        paths = find_tool_paths(data, "google_search")
                        
                        # Print the first 5 locations it found the tool
                        for p in paths[:5]:
                            print(p)
                            print("-" * 50)
                        exit()
            except Exception as e:
                pass