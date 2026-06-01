#!/usr/bin/env python3
"""Parse training logs and print a results table."""
import argparse
import glob
import os
import re

METRICS = ("Overall", "Many", "Median", "Low")


def parse_log(path):
    with open(path) as f:
        text = f.read()
    blocks = list(re.finditer(r"\* Test Overall:.*", text, re.DOTALL))
    if not blocks:
        blocks = list(re.finditer(r"\* Val Overall:.*", text, re.DOTALL))
    if not blocks:
        return None
    section = blocks[-1].group(0)
    rows = {}
    for name in METRICS:
        m = re.search(
            rf"\* (?:Test|Val) {name}:.*?L1\s+([\d.]+|nan).*?G-Mean\s+([\d.]+|nan)",
            section,
        )
        if m:
            rows[name] = {"l1": m.group(1), "gmean": m.group(2)}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default="checkpoint")
    parser.add_argument("--filter", default="", help="substring filter on run name")
    args = parser.parse_args()

    print(f"{'Run':<60} {'Overall L1':>10} {'Low L1':>10} {'Low G-Mean':>10}")
    print("-" * 95)
    for log_path in sorted(glob.glob(os.path.join(args.checkpoint_root, "*/training.log"))):
        name = os.path.basename(os.path.dirname(log_path))
        if args.filter and args.filter not in name:
            continue
        rows = parse_log(log_path)
        if not rows:
            print(f"{name:<60} {'(no metrics)':>10}")
            continue
        overall = rows.get("Overall", {}).get("l1", "-")
        low_l1 = rows.get("Low", {}).get("l1", "-")
        low_gm = rows.get("Low", {}).get("gmean", "-")
        print(f"{name:<60} {overall:>10} {low_l1:>10} {low_gm:>10}")


if __name__ == "__main__":
    main()
