#!/usr/bin/env python3
"""Parse training logs and print a results table."""
import argparse
import glob
import os
import re

METRICS = ("Overall", "Many", "Median", "Low")


def parse_log(path):
    rows = {}
    with open(path) as f:
        text = f.read()
    blocks = list(re.finditer(r"\* Test Overall:.*", text, re.DOTALL))
    if not blocks:
        return None
    section = blocks[-1].group(0)
    for name in METRICS:
        m = re.search(
            rf"\* Test {name}:.*?L1\s+([\d.]+|nan).*?G-Mean\s+([\d.]+|nan)",
            section,
        )
        if m:
            rows[name] = {"l1": m.group(1), "gmean": m.group(2)}
    return rows if rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default="checkpoint")
    args = parser.parse_args()

    print(f"{'Run':<55} {'Overall L1':>10} {'Low L1':>10} {'Low G-Mean':>10}")
    print("-" * 90)
    for log_path in sorted(glob.glob(os.path.join(args.checkpoint_root, "*/training.log"))):
        name = os.path.basename(os.path.dirname(log_path))
        rows = parse_log(log_path)
        if not rows:
            print(f"{name:<55} {'(no test metrics)':>10}")
            continue
        overall = rows.get("Overall", {}).get("l1", "-")
        low_l1 = rows.get("Low", {}).get("l1", "-")
        low_gm = rows.get("Low", {}).get("gmean", "-")
        print(f"{name:<55} {overall:>10} {low_l1:>10} {low_gm:>10}")


if __name__ == "__main__":
    main()
