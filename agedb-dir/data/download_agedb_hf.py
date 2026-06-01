#!/usr/bin/env python3
"""
Download AgeDB images from Hugging Face (34data/AgeDB) into data/AgeDB/
for use with bundled agedb.csv paths (AgeDB/<filename>).
"""
import argparse
import os
import sys

# Run from repo root or data/; avoid local datasets.py shadowing HF
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)
# Remove agedb-dir from path if it shadows huggingface datasets
_agedb_root = os.path.dirname(_DATA_DIR)
if _agedb_root in sys.path:
    sys.path.remove(_agedb_root)

import pandas as pd
from io import BytesIO
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--csv", default="agedb.csv")
    parser.add_argument("--limit", type=int, default=0, help="max images (0=all in csv)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets pillow")
        sys.exit(1)

    csv_path = os.path.join(args.data_dir, args.csv)
    df = pd.read_csv(csv_path)
    needed = set(df["path"].map(lambda p: os.path.basename(p)))
    if args.limit:
        needed = set(list(needed)[: args.limit])

    out_dir = os.path.join(args.data_dir, "AgeDB")
    os.makedirs(out_dir, exist_ok=True)
    existing = set(os.listdir(out_dir))
    todo = needed - existing
    print(f"Need {len(todo)} / {len(needed)} images ({len(existing)} already on disk)")

    if not todo:
        print("Done.")
        return

    ds = load_dataset("34data/AgeDB", split="train")
    saved = 0
    for row in ds:
        fname = row["filename"]
        if fname not in todo:
            continue
        dest = os.path.join(out_dir, fname)
        img = row["image"]
        if isinstance(img, bytes):
            img = Image.open(BytesIO(img)).convert("RGB")
        elif not hasattr(img, "save"):
            img = Image.open(BytesIO(img["bytes"])).convert("RGB")
        img.save(dest, format="JPEG")
        saved += 1
        if saved % 500 == 0:
            print(f"Saved {saved}/{len(todo)}")
        if saved >= len(todo):
            break

    print(f"Saved {saved} images to {out_dir}")


if __name__ == "__main__":
    main()
