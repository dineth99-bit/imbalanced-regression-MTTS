#!/usr/bin/env python3
"""Print temperature bin counts per split (for tuning LDS/FDS / shot thresholds)."""
import argparse
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(BASE, "skyfinder.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["temp_int"] = df["temperature"].round().astype(int)

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        vc = sub["temp_int"].value_counts().sort_index()
        print(f"\n=== {split.upper()} ({len(sub)} images, {len(vc)} bins) ===")
        print(f"  count per bin: min={vc.min()}, median={vc.median():.0f}, max={vc.max()}")
        for thr in [3, 5, 10, 20, 50]:
            n_low = (vc < thr).sum()
            print(f"  bins with <{thr} samples: {n_low}/{len(vc)}")
        print("  (first 15 bins)")
        for t, c in list(vc.items())[:15]:
            print(f"    {t}°C: {c}")
        if len(vc) > 15:
            print("    ...")

    train_vc = df[df["split"] == "train"]["temp_int"].value_counts()
    print("\nSuggested shot thresholds for train.py:")
    print(f"  --many_shot_thr {max(10, int(train_vc.median()))}")
    print(f"  --low_shot_thr {max(3, int(train_vc.quantile(0.25)))}")


if __name__ == "__main__":
    main()
