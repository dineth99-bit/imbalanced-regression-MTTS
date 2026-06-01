"""
Small hyperparameter grid for SkyFinder-DIR. Runs train.py, parses test metrics from logs.

Usage:
  TUNE_EPOCHS=15 python tune_hyperparams.py
  TUNE_EPOCHS=30 python tune_hyperparams.py --dry-run
"""
import argparse
import os
import re
import subprocess
import sys
from itertools import product

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(ROOT, "train.py")


def parse_test_metrics(log_path):
    if not os.path.isfile(log_path):
        return None
    text = open(log_path).read()
    blocks = list(re.finditer(r"\* Test Overall:.*", text, re.DOTALL))
    if not blocks:
        return None
    section = blocks[-1].group(0)

    def grab(name):
        m = re.search(
            rf"\* Test {name}:.*?L1\s+([\d.]+|nan).*?(?:G-Mean\s+([\d.]+|nan))?",
            section,
        )
        if not m:
            return None, None
        l1 = float("nan") if m.group(1) == "nan" else float(m.group(1))
        gm = float("nan")
        if m.lastindex and m.lastindex >= 2 and m.group(2) and m.group(2) != "nan":
            gm = float(m.group(2))
        return l1, gm

    overall_l1, overall_gm = grab("Overall")
    low_l1, low_gm = grab("Low")
    med_l1, _ = grab("Median")
    return {
        "overall_l1": overall_l1,
        "overall_gmean": overall_gm,
        "median_l1": med_l1,
        "low_l1": low_l1,
        "low_gmean": low_gm,
    }


def build_grid():
    """Focused grid for small temperature-regression data."""
    configs = []
    base = {
        "reweight": "none",
        "lds": False,
        "fds": False,
        "lds_ks": 5,
        "lds_sigma": 2,
        "fds_ks": 5,
        "fds_sigma": 2,
        "start_update": 0,
        "start_smooth": 1,
        "label_min": -15,
        "label_max": 40,
        "bucket_num": 56,
        "many_shot_thr": 20,
        "low_shot_thr": 5,
    }
    configs.append({**base, "name": "vanilla"})

    for rw in ["sqrt_inv"]:
        configs.append({**base, "name": f"{rw}", "reweight": rw})

    for sigma in [1, 2, 3]:
        configs.append({
            **base,
            "name": f"lds_s{sigma}",
            "reweight": "sqrt_inv",
            "lds": True,
            "lds_sigma": sigma,
        })

    for start_smooth in [1, 5, 10]:
        configs.append({
            **base,
            "name": f"fds_smooth{start_smooth}",
            "fds": True,
            "start_smooth": start_smooth,
            "start_update": min(start_smooth, 5),
        })

    for sigma in [1, 2]:
        for start_smooth in [5]:
            configs.append({
                **base,
                "name": f"lds_fds_s{sigma}_sm{start_smooth}",
                "reweight": "sqrt_inv",
                "lds": True,
                "fds": True,
                "lds_sigma": sigma,
                "fds_sigma": sigma,
                "start_smooth": start_smooth,
                "start_update": 5,
            })

    return configs


def config_to_cmd(cfg, epochs, batch_size, workers):
    cmd = [
        sys.executable, TRAIN,
        "--meta_dir", os.path.join(ROOT, "data"),
        "--data_dir", os.path.join(ROOT, "data", "images"),
        "--mask_dir", os.path.join(ROOT, "data"),
        "--epoch", str(epochs),
        "--batch_size", str(batch_size),
        "--workers", str(workers),
        "--schedule", "20", "25",
        "--store_name", f"tune_{cfg['name']}",
        "--reweight", cfg["reweight"],
        "--lds_ks", str(cfg["lds_ks"]),
        "--lds_sigma", str(cfg["lds_sigma"]),
        "--fds_ks", str(cfg["fds_ks"]),
        "--fds_sigma", str(cfg["fds_sigma"]),
        "--start_update", str(cfg["start_update"]),
        "--start_smooth", str(cfg["start_smooth"]),
        "--label_min", str(cfg["label_min"]),
        "--label_max", str(cfg["label_max"]),
        "--bucket_num", str(cfg["bucket_num"]),
        "--many_shot_thr", str(cfg["many_shot_thr"]),
        "--low_shot_thr", str(cfg["low_shot_thr"]),
    ]
    if cfg.get("lds"):
        cmd.append("--lds")
    if cfg.get("fds"):
        cmd.append("--fds")
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("TUNE_EPOCHS", "15")))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default=os.path.join(ROOT, "tune_results.csv"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="max configs (0=all)")
    args = parser.parse_args()

    configs = build_grid()
    if args.limit:
        configs = configs[: args.limit]

    rows = []
    for i, cfg in enumerate(configs):
        cmd = config_to_cmd(cfg, args.epochs, args.batch_size, args.workers)
        print(f"\n[{i+1}/{len(configs)}] {cfg['name']}")
        print(" ", " ".join(cmd))
        if args.dry_run:
            continue

        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

        subprocess.run(cmd, cwd=ROOT, check=True, env=env)

        store = f"tune_{cfg['name']}_resnet50"
        if cfg["reweight"] != "none" and not cfg.get("lds"):
            store += f"_{cfg['reweight']}"
        # store_name gets more suffixes from train.py; find log by glob
        import glob
        matches = glob.glob(os.path.join(ROOT, "checkpoint", f"*{cfg['name']}*", "training.log"))
        log_path = matches[0] if matches else ""
        metrics = parse_test_metrics(log_path) or {}
        row = {**cfg, **metrics, "log": log_path}
        rows.append(row)

    if args.dry_run:
        print(f"\n{len(configs)} configs (dry-run).")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")

    if len(df):
        for col in ["low_l1", "overall_l1"]:
            if col in df.columns:
                print(f"\nTop 5 by {col}:")
                print(df.sort_values(col).head(5)[["name", "overall_l1", "low_l1", "low_gmean"]].to_string(index=False))

        best = df.sort_values("overall_l1").iloc[0]
        best_low = df.sort_values("low_l1").iloc[0]
        print("\n--- Best picks ---")
        print(f"Best overall: {best['name']}  overall_l1={best['overall_l1']:.3f}  low_l1={best['low_l1']:.3f}")
        print(f"Best low-shot: {best_low['name']}  overall_l1={best_low['overall_l1']:.3f}  low_l1={best_low['low_l1']:.3f}")
        print("Retrain: bash train_best.sh  (or copy flags from tune_results.csv row)")


if __name__ == "__main__":
    main()
