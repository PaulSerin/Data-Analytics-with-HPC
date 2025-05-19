# print_paths.py

import os
import sys
from pathlib import Path

def main():
    cwd = Path.cwd()
    # __file__ may be undefined if run interactively; fallback to argv[0]
    script_path = (Path(__file__) if "__file__" in globals() else Path(sys.argv[0])).resolve()
    print(f"Current working directory: {cwd}")
    print(f"Script path: {script_path}\n")

    print("Python sys.path:")
    for p in sys.path:
        print(f"  {p}")
    print()

    # Files to verify relative to cwd
    files_to_check = [
        "Datasets/aus_open_2025_matches_all_ids.json",
        "Datasets/final_tennis_dataset_symmetric.parquet",
        "Models/xgb_model.json",
        "../0.Utils/utils.py"
    ]
    print("File existence checks:")
    for rel in files_to_check:
        p = cwd / rel
        status = "Exists" if p.exists() else "Missing"
        print(f"  {rel}: {status} (checked at {p})")

if __name__ == "__main__":
    main()
