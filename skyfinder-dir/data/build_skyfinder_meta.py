"""
Build skyfinder_raw.csv from on-disk images + official SkyFinder metadata (TempM).

Usage (from skyfinder-dir/data):
  python build_skyfinder_meta.py --images_dir ./images --metadata_csv ./skyfinder_metadata.csv
"""
import argparse
import os

import pandas as pd

MISSING_TEMP = -9999


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="./images")
    parser.add_argument("--metadata_csv", default="./skyfinder_metadata.csv")
    parser.add_argument("--output", default="./skyfinder_raw.csv")
    parser.add_argument("--hour_min", type=int, default=10)
    parser.add_argument("--hour_max", type=int, default=12)
    parser.add_argument("--cameras", nargs="*", type=int, default=None)
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata_csv)
    meta = meta.rename(columns={"CamId": "camera_id", "Filename": "filename", "TempM": "temperature"})
    meta["path"] = meta["camera_id"].astype(str) + "/" + meta["filename"].astype(str)
    meta = meta[(meta["Hour"] >= args.hour_min) & (meta["Hour"] <= args.hour_max)]
    meta = meta[meta["temperature"] > MISSING_TEMP + 1]

    if args.cameras:
        meta = meta[meta["camera_id"].isin(args.cameras)]

    on_disk = set()
    for cam in os.listdir(args.images_dir):
        cam_dir = os.path.join(args.images_dir, cam)
        if not os.path.isdir(cam_dir):
            continue
        for fname in os.listdir(cam_dir):
            if fname.lower().endswith(".jpg"):
                on_disk.add(f"{cam}/{fname}")

    before = len(meta)
    meta = meta[meta["path"].isin(on_disk)]
    print(f"Matched {len(meta)} / {before} metadata rows to files under {args.images_dir}")

    meta["timestamp"] = meta.apply(
        lambda r: (
            f"{int(r['Year']):04d}-{int(r['Month']):02d}-{int(r['Day']):02d} "
            f"{int(r['Hour']):02d}:{int(r['Min']):02d}:00"
        ),
        axis=1,
    )
    out = meta[["temperature", "path", "camera_id", "timestamp"]].copy()
    out["temperature"] = out["temperature"].astype(float).round(1)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print(f"Temperature range: [{out['temperature'].min():.1f}, {out['temperature'].max():.1f}]")
    print(f"Cameras: {sorted(out['camera_id'].unique().tolist())}")


if __name__ == "__main__":
    main()
