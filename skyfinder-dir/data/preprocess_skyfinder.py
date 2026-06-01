"""
Create skyfinder.csv with train/val/test splits for SkyFinder-DIR.

- Val/test cameras are disjoint from train cameras (default).
- Val/test images: balanced caps per integer temperature (DIR protocol).
- Train: all images from train cameras + unused images from val/test cameras.
"""
import argparse
import os
import random

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def make_balanced_subset(df, split_name, max_per_bin=25, seed=666):
    """Cap samples per integer temperature bin for one split (val or test)."""
    random.seed(seed)
    chosen = []
    temps = df["temperature"].astype(int)
    for value in sorted(temps.unique()):
        curr = df[temps == value]["path"].tolist()
        random.shuffle(curr)
        n = min(len(curr), max_per_bin)
        chosen.extend(curr[:n])
    return {p: split_name for p in chosen}


def make_balanced_val_test(df, max_per_bin=25, seed=666):
    """Random-split DIR: balanced val and test from the full set."""
    random.seed(seed)
    val_paths, test_paths = [], []
    temps = df["temperature"].astype(int)
    for value in sorted(temps.unique()):
        curr = df[temps == value]["path"].tolist()
        random.shuffle(curr)
        n = min(len(curr) // 3, max_per_bin)
        val_paths.extend(curr[:n])
        test_paths.extend(curr[n:n * 2])
    combined = {p: "val" for p in val_paths}
    combined.update({p: "test" for p in test_paths})
    return combined


def camera_disjoint_split(df, val_ratio=0.15, test_ratio=0.15, seed=666):
    random.seed(seed)
    cameras = sorted(df["camera_id"].unique().tolist())
    random.shuffle(cameras)
    n_test = max(1, int(len(cameras) * test_ratio))
    n_val = max(1, int(len(cameras) * val_ratio))
    test_cams = set(cameras[:n_test])
    val_cams = set(cameras[n_test:n_test + n_val])
    train_cams = set(cameras[n_test + n_val:])
    return train_cams, val_cams, test_cams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(BASE, "skyfinder_raw.csv"))
    parser.add_argument("--output", default=os.path.join(BASE, "skyfinder.csv"))
    parser.add_argument("--max_per_bin", type=int, default=25)
    parser.add_argument("--random_split", action="store_true")
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # drop rows whose images fail to load (truncated downloads)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    valid = []
    for _, row in df.iterrows():
        p = os.path.join(BASE, "images", row["path"])
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.convert("RGB").load()
            valid.append(True)
        except Exception:
            valid.append(False)
    dropped = (~pd.Series(valid)).sum()
    if dropped:
        print(f"Dropping {dropped} rows with unreadable images")
        df = df[valid].reset_index(drop=True)
    split = {p: "train" for p in df["path"]}

    if args.random_split:
        random.seed(args.seed)
        balanced = make_balanced_val_test(df, max_per_bin=args.max_per_bin, seed=args.seed)
        split.update(balanced)
        print("Random split with balanced val/test per temperature bin")
    else:
        train_cams, val_cams, test_cams = camera_disjoint_split(df, seed=args.seed)
        print(f"Train cameras: {sorted(train_cams)}")
        print(f"Val cameras:   {sorted(val_cams)}")
        print(f"Test cameras:  {sorted(test_cams)}")

        val_df = df[df["camera_id"].isin(val_cams)]
        test_df = df[df["camera_id"].isin(test_cams)]
        val_balanced = make_balanced_subset(val_df, "val", max_per_bin=args.max_per_bin, seed=args.seed)
        test_balanced = make_balanced_subset(test_df, "test", max_per_bin=args.max_per_bin, seed=args.seed + 1)
        split.update(val_balanced)
        split.update(test_balanced)

    df["split"] = df["path"].map(split)
    out = df[["temperature", "path", "split", "camera_id"]]
    out.to_csv(args.output, index=False)
    for sp in ["train", "val", "test"]:
        sub = out[out["split"] == sp]
        print(f"{sp}: {len(sub)} images, cameras {sorted(sub['camera_id'].unique())}")


if __name__ == "__main__":
    main()
