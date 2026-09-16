#!/usr/bin/env python
"""Compare RF-DETR ablation results across variants.

Reads metrics.csv from each variant's output dir and prints a summary table.
"""
import csv
import os
import sys
from pathlib import Path

OUTPUT_ROOT = Path("outputs/mot")
VARIANTS = [
    "rfdetr-baseline",
    "rfdetr-small-baseline",
    "rfdetr-small-no-dn",
    "rfdetr-small-50q",
    "rfdetr-small-2layer",
    "rfdetr-small-no-pretrain",
]


def read_best_metrics(variant_dir: Path) -> dict:
    """Read metrics.csv and return best epoch metrics."""
    csv_path = variant_dir / "metrics.csv"
    if not csv_path.exists():
        return {}

    best = {}
    best_map = -1
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # RF-DETR logs metrics in columns; find the relevant ones
            for key in row:
                if key is None:
                    continue
                # Try to parse as float
                try:
                    val = float(row[key])
                except (ValueError, TypeError):
                    continue

                # Track best mAP
                if "map" in key.lower() and "50" not in key.lower() and "75" not in key.lower():
                    if val > best_map:
                        best_map = val
                        best["best_mAP"] = val
                        best["best_epoch"] = int(float(row.get("epoch", row.get("step", 0))))

                if "map_50" in key.lower() or ("map50" in key.lower()):
                    best["best_mAP50"] = max(best.get("best_mAP50", 0), val)
                if "map_75" in key.lower() or ("map75" in key.lower()):
                    best["best_mAP75"] = max(best.get("best_mAP75", 0), val)

    return best


def main():
    print(f"{'Variant':<30} {'Epoch':>6} {'mAP@50:95':>10} {'mAP@50':>10} {'mAP@75':>10}")
    print("-" * 70)

    for name in VARIANTS:
        vdir = OUTPUT_ROOT / name
        if not vdir.exists():
            print(f"{name:<30} {'—':>6} {'—':>10} {'—':>10} {'—':>10}")
            continue

        metrics = read_best_metrics(vdir)
        if not metrics:
            print(f"{name:<30} {'?':>6} {'?':>10} {'?':>10} {'?':>10}")
            continue

        print(
            f"{name:<30} "
            f"{metrics.get('best_epoch', '?'):>6} "
            f"{metrics.get('best_mAP', 0):>10.4f} "
            f"{metrics.get('best_mAP50', 0):>10.4f} "
            f"{metrics.get('best_mAP75', 0):>10.4f}"
        )


if __name__ == "__main__":
    main()
