#!/usr/bin/env python3
"""Verify AgeDB images exist for paths in agedb.csv."""
import argparse
import os
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--csv", default="agedb.csv")
    parser.add_argument("--min_found", type=int, default=100)
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, args.csv)
    df = pd.read_csv(csv_path)
    found = 0
    missing = 0
    for path in df["path"].head(500):
        full = os.path.join(args.data_dir, path)
        if os.path.isfile(full):
            found += 1
        else:
            missing += 1
    total_check = found + missing
    print(f"Checked {total_check} paths: {found} found, {missing} missing under {args.data_dir}")
    if found < args.min_found:
        print(f"Need at least {args.min_found} images; extract AgeDB zip to {args.data_dir}/AgeDB/")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
