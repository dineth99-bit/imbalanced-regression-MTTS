#!/usr/bin/env python3
"""Keep only rows in agedb.csv whose images exist under data_dir."""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--input", default="agedb.csv")
    parser.add_argument("--output", default="agedb.csv")
    args = parser.parse_args()

    path = os.path.join(args.data_dir, args.input)
    df = pd.read_csv(path)
    before = len(df)
    exists = df["path"].apply(lambda p: os.path.isfile(os.path.join(args.data_dir, p)))
    df = df[exists].reset_index(drop=True)
    out = os.path.join(args.data_dir, args.output)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} / {before} rows to {out}")
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        print(f"  {sp}: {len(sub)}")


if __name__ == "__main__":
    main()
