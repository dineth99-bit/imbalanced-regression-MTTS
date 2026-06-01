#!/usr/bin/env python3
"""Download SkyFinder camera zips and masks from Zenodo."""
import argparse
import os
import subprocess

ZENODO = "https://zenodo.org/records/5884485/files"
# Cameras with rows in skyfinder_metadata.csv (replaces 858 which has no official weather)
CAMERAS_5 = ["1093", "3888", "4795", "5021", "3297"]
# +5 high-coverage cameras for 10-camera benchmark
CAMERAS_10_EXTRA = ["684", "623", "3837", "5020", "65"]
DEFAULT_CAMERAS = CAMERAS_5 + CAMERAS_10_EXTRA


def download(url, dest):
    if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
        print(f"Skip existing {dest}")
        return
    print(f"Downloading {url}")
    subprocess.check_call(["curl", "-L", "-o", dest, url])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--cameras", nargs="*", default=DEFAULT_CAMERAS)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    download(
        f"{ZENODO}/skyfinder_masks.zip?download=1",
        os.path.join(args.data_dir, "skyfinder_masks.zip"),
    )
    masks_dir = os.path.join(args.data_dir, "masks", "skyfinder_masks")
    if not os.path.isdir(masks_dir):
        subprocess.check_call(
            ["unzip", "-q", "-o", os.path.join(args.data_dir, "skyfinder_masks.zip"),
             "-d", os.path.join(args.data_dir, "masks")],
            cwd=args.data_dir,
        )

    for cam in args.cameras:
        download(f"{ZENODO}/{cam}.zip?download=1", os.path.join(args.data_dir, f"{cam}.zip"))
        img_root = os.path.join(args.data_dir, "images")
        os.makedirs(img_root, exist_ok=True)
        subprocess.check_call(
            ["unzip", "-q", "-o", os.path.join(args.data_dir, f"{cam}.zip"), "-d", img_root],
            cwd=args.data_dir,
        )

    print("Done. Next:")
    print(f"  cd {os.path.abspath(args.data_dir)} && python build_skyfinder_meta.py")
    print("  python preprocess_skyfinder.py")


if __name__ == "__main__":
    main()
