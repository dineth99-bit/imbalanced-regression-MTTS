#!/usr/bin/env python3
"""Build RESULTS.md from experiments_manifest.csv and training logs."""
import argparse
import csv
import glob
import os
import re
from datetime import datetime

METRICS = ("Overall", "Many", "Median", "Low")


def parse_log(path):
    if not os.path.isfile(path):
        return None
    text = open(path).read()
    # Prefer final test eval; fall back to last val block (validate() prints "* Overall:" not "* Test Overall:")
    test_idx = text.rfind("Test best model")
    search_text = text[test_idx:] if test_idx >= 0 else text
    blocks = list(re.finditer(r"\* Overall:.*", search_text, re.DOTALL))
    if not blocks:
        return None
    section = blocks[-1].group(0)
    out = {}
    for name in METRICS:
        m = re.search(
            rf"\* {name}: MSE [\d.]+\s+L1 ([\d.]+|nan)\s+G-Mean ([\d.]+|nan)",
            section,
        )
        if m:
            out[name] = {"l1": m.group(1), "gmean": m.group(2)}
    return out


def find_log(store_substr, checkpoint_root):
    candidates = []
    for log_path in glob.glob(os.path.join(checkpoint_root, "*/training.log")):
        folder = os.path.basename(os.path.dirname(log_path))
        if f"_{store_substr}_" in folder or folder.endswith(f"_{store_substr}"):
            candidates.append((len(folder), log_path))
    if not candidates:
        return None
    return min(candidates)[1]


def fmt(x):
    if x is None or x == "-" or str(x).lower() == "nan":
        return "—"
    try:
        return f"{float(x):.2f}"
    except ValueError:
        return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="experiments_manifest.csv")
    parser.add_argument("--checkpoint_root", default="checkpoint")
    parser.add_argument("--output", default="RESULTS.md")
    args = parser.parse_args()

    rows = []
    with open(args.manifest) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    table = []
    best_mtts = None
    vanilla = None

    for r in rows:
        method = r["method"]
        status = r.get("status", "")
        store = r.get("store_substr", "")

        if status == "SKIPPED":
            table.append((method, "SKIPPED", "—", "—", "—", "—", "—", "—", "—"))
            continue
        if status == "FAILED":
            table.append((method, "FAILED", "—", "—", "—", "—", "—", "—", "—"))
            continue

        log_path = find_log(store, args.checkpoint_root)
        metrics = parse_log(log_path) if log_path else None
        if not metrics:
            table.append((method, "NO_LOG", "—", "—", "—", "—", "—", "—", "—"))
            continue

        o, m, med, lo = metrics.get("Overall", {}), metrics.get("Many", {}), metrics.get("Median", {}), metrics.get("Low", {})
        table.append((
            method, status,
            fmt(o.get("l1")), fmt(m.get("l1")), fmt(med.get("l1")), fmt(lo.get("l1")),
            fmt(o.get("gmean")), fmt(m.get("gmean")), fmt(med.get("gmean")), fmt(lo.get("gmean")),
        ))
        if method == "VANILLA" and o.get("l1"):
            vanilla = metrics
        if "MTTS" in method and "no meta" not in method.lower() and "unconstrained" not in method.lower():
            if best_mtts is None or (lo.get("l1") and float(lo["l1"]) < float(best_mtts[1].get("l1", 999))):
                best_mtts = (method, lo, o)

    lines = [
        "# AgeDB-DIR Full Benchmark Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "**Setup:** ResNet50, 90 epochs, L1 (FOCAL-R rows use `focal_l1`), LDS/FDS Gaussian ks=5 σ=2.",
        f"**Data:** `data/agedb.csv` (on-disk images). **Train:** {sum(1 for r in rows if r.get('status')=='DONE')} runs completed.",
        "",
        "**Note:** SMOTER / SMOGN rows skipped (not implemented in this repo).",
        "",
        "## Results table (Test set)",
        "",
        "| Method | MAE All | MAE Many | MAE Med. | MAE Few | GM All | GM Many | GM Med. | GM Few |",
        "|--------|--------:|---------:|---------:|--------:|-------:|--------:|--------:|-------:|",
    ]

    for row in table:
        method = row[0]
        if row[1] in ("SKIPPED", "FAILED", "NO_LOG"):
            lines.append(f"| {method} | *{row[1]}* | — | — | — | — | — | — | — |")
        else:
            lines.append(f"| {row[0]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} | {row[9]} |")

    if vanilla and best_mtts:
        vo = vanilla["Overall"]["l1"]
        vm = vanilla["Many"]["l1"]
        vmed = vanilla["Median"]["l1"]
        vlo = vanilla["Low"]["l1"]
        vgo = vanilla["Overall"]["gmean"]
        vgm = vanilla["Many"]["gmean"]
        vgmed = vanilla["Median"]["gmean"]
        vglo = vanilla["Low"]["gmean"]
        bo, blo, bglo = best_mtts[2]["l1"], best_mtts[1]["l1"], best_mtts[2]["gmean"]
        lines.extend([
            "",
            "## MTTS (BEST) vs VANILLA (Test MAE / GM deltas)",
            "",
            f"Best MTTS config: **{best_mtts[0]}**",
            "",
            "| Shot | Δ MAE (Vanilla − Best MTTS) | Δ GM (Vanilla − Best MTTS) |",
            "|------|---------------------------:|---------------------------:|",
            f"| All | {float(vo)-float(bo):+.2f} | {float(vgo)-float(bglo):+.2f} |",
            f"| Many | {float(vm)-float(vanilla['Many']['l1']):+.2f} | {float(vgm)-float(vanilla['Many']['gmean']):+.2f} |",
            f"| Med. | {float(vmed)-float(vanilla['Median']['l1']):+.2f} | {float(vgmed)-float(vanilla['Median']['gmean']):+.2f} |",
            f"| Few | {float(vlo)-float(blo):+.2f} | {float(vglo)-float(best_mtts[1]['gmean']):+.2f} |",
        ])

    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd agedb-dir",
        "CUDA_VISIBLE_DEVICES=0 bash run_full_benchmark.sh",
        "python3 collect_results.py",
        "```",
        "",
    ])

    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
