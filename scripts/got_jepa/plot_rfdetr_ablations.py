#!/usr/bin/env python3
"""Generate comparison plots from RF-DETR ablation study results.

Reads metrics.csv from each variant's output directory and produces:
1. EMA mAP@50:95 over epochs (line plot, all variants)
2. Regular mAP@50:95 over epochs (line plot, all variants)
3. Best mAP bar chart (EMA vs regular, all variants)
4. Per-class AP comparison (grouped bar chart)
5. Validation loss over epochs
6. mAP@50 over epochs
7. Precision vs Recall scatter
8. Delta from baseline (waterfall chart)

Usage:
    python scripts/got_jepa/plot_rfdetr_ablations.py [--output-dir outputs/mot/plots]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Variant definitions
# ──────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    ("baseline",    "rfdetr-small-baseline",    "Baseline (full)",          "#2196F3"),
    ("no-dn",       "rfdetr-small-no-dn",       "No denoising",             "#FF9800"),
    ("50q",         "rfdetr-small-50q",         "50 queries",               "#4CAF50"),
    ("2layer",      "rfdetr-small-2layer",      "2 decoder layers",         "#9C27B0"),
    ("no-pretrain", "rfdetr-small-no-pretrain", "No pretrain",              "#F44336"),
    ("no-ema",      "rfdetr-small-no-ema",      "No EMA",                   "#795548"),
    ("highlr",      "rfdetr-small-highlr",      "2× LR",                    "#00BCD4"),
]

CLASSES = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]
CLASS_COLORS = ["#e57373", "#81c784", "#64b5f6", "#ffb74d", "#ba68c8", "#4db6ac"]

OUTPUT_BASE = Path("outputs/mot")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_variant_csv(dirname: str) -> list[dict]:
    """Load metrics.csv for a variant."""
    path = OUTPUT_BASE / dirname / "metrics.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_series(rows: list[dict], key: str) -> tuple[list, list]:
    """Extract (epochs, values) for a given metric key."""
    epochs, vals = [], []
    for r in rows:
        try:
            ep = int(float(r["epoch"]))
            v = r.get(key, "")
            if v and v.strip():
                epochs.append(ep)
                vals.append(float(v))
        except (ValueError, KeyError):
            continue
    return epochs, vals


def get_best(rows: list[dict], key: str, skip_ema_anomaly: bool = False) -> tuple[float, int]:
    """Get best value and epoch for a metric key."""
    best_v, best_ep = 0.0, 0
    for r in rows:
        try:
            ep = int(float(r["epoch"]))
            v = float(r.get(key, "") or 0)
            if skip_ema_anomaly and v > 0.6:
                continue
            if v > best_v:
                best_v, best_ep = v, ep
        except (ValueError, KeyError):
            continue
    return best_v, best_ep


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 9,
    })


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved: {out_dir / name}.png + .pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Individual plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_ema_map_over_epochs(data, out_dir):
    """EMA mAP@50:95 over epochs for all variants."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, dirname, display, color in VARIANTS:
        rows = data.get(dirname, [])
        if not rows:
            continue
        eps, vals = extract_series(rows, "val/ema_mAP_50_95")
        if not vals:
            continue
        # Skip no-pretrain anomaly (values > 0.6 are corrupted)
        if label == "no-pretrain":
            vals_filt = [v for v in vals if v < 0.6]
            eps_filt = [e for e, v in zip(eps, vals) if v < 0.6]
            if eps_filt:
                ax.plot(eps_filt, vals_filt, color=color, label=display, linewidth=2, alpha=0.8)
            continue
        if label == "no-ema":
            continue  # No EMA columns
        ax.plot(eps, vals, color=color, label=display, linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("EMA mAP@50:95")
    ax.set_title("EMA mAP@50:95 Over Training Epochs")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 30)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    save_fig(fig, out_dir, "ema_map_50_95_over_epochs")


def plot_reg_map_over_epochs(data, out_dir):
    """Regular mAP@50:95 over epochs for all variants."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, dirname, display, color in VARIANTS:
        rows = data.get(dirname, [])
        if not rows:
            continue
        eps, vals = extract_series(rows, "val/mAP_50_95")
        if not vals:
            continue
        ax.plot(eps, vals, color=color, label=display, linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Regular mAP@50:95")
    ax.set_title("Regular mAP@50:95 Over Training Epochs")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 30)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    save_fig(fig, out_dir, "reg_map_50_95_over_epochs")


def plot_best_map_bar(data, out_dir):
    """Best mAP bar chart comparing EMA vs regular for all variants."""
    labels = []
    reg_bests = []
    ema_bests = []
    colors = []
    for label, dirname, display, color in VARIANTS:
        rows = data.get(dirname, [])
        if not rows:
            continue
        reg_v, _ = get_best(rows, "val/mAP_50_95")
        ema_v, _ = get_best(rows, "val/ema_mAP_50_95", skip_ema_anomaly=(label == "no-pretrain"))
        labels.append(display)
        reg_bests.append(reg_v)
        ema_bests.append(ema_v if ema_v > 0 else 0)
        colors.append(color)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, reg_bests, width, label="Regular mAP@50:95", color=[c + "99" for c in colors], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, ema_bests, width, label="EMA mAP@50:95", color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_ylabel("Best mAP@50:95")
    ax.set_title("Best mAP@50:95 — Regular vs EMA by Variant")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(reg_bests), max(ema_bests)) * 1.25)
    save_fig(fig, out_dir, "best_map_comparison_bars")


def plot_per_class_ap(data, out_dir):
    """Per-class AP comparison (grouped bar chart) at best EMA epoch."""
    n_variants = len(VARIANTS)
    n_classes = len(CLASSES)
    width = 0.8 / n_variants
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (label, dirname, display, color) in enumerate(VARIANTS):
        rows = data.get(dirname, [])
        if not rows:
            continue
        # Find best EMA epoch (or best regular for no-ema)
        if label != "no-ema":
            _, best_ep = get_best(rows, "val/ema_mAP_50_95", skip_ema_anomaly=(label == "no-pretrain"))
        else:
            _, best_ep = get_best(rows, "val/mAP_50_95")

        # Get per-class AP at that epoch
        class_aps = []
        for cls in CLASSES:
            key = f"val/AP/{cls}"
            val = 0.0
            for r in rows:
                try:
                    if int(float(r["epoch"])) == best_ep:
                        val = float(r.get(key, "") or 0)
                        break
                except:
                    continue
            class_aps.append(val)

        x = np.arange(n_classes)
        ax.bar(x + i * width - 0.4 + width/2, class_aps, width, label=display, color=color, alpha=0.85)

    ax.set_xlabel("Class")
    ax.set_ylabel("AP@50:95")
    ax.set_title("Per-class AP@50:95 at Best Epoch (by Variant)")
    ax.set_xticks(np.arange(n_classes))
    ax.set_xticklabels(CLASSES, rotation=20)
    ax.legend(loc="upper right", ncol=2)
    save_fig(fig, out_dir, "per_class_ap_bars")


def plot_val_loss(data, out_dir):
    """Validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, dirname, display, color in VARIANTS:
        rows = data.get(dirname, [])
        if not rows:
            continue
        eps, vals = extract_series(rows, "val/loss")
        if not vals:
            continue
        ax.plot(eps, vals, color=color, label=display, linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Over Training Epochs")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 30)
    save_fig(fig, out_dir, "val_loss_over_epochs")


def plot_map50_over_epochs(data, out_dir):
    """mAP@50 over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, dirname, display, color in VARIANTS:
        rows = data.get(dirname, [])
        if not rows:
            continue
        eps, vals = extract_series(rows, "val/mAP_50")
        if not vals:
            continue
        ax.plot(eps, vals, color=color, label=display, linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50")
    ax.set_title("mAP@50 Over Training Epochs")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 30)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    save_fig(fig, out_dir, "map50_over_epochs")


def plot_precision_recall_scatter(data, out_dir):
    """Precision vs Recall scatter at best epoch for each variant."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for label, dirname, display, color in VARIANTS:
        rows = data.get(dirname, [])
        if not rows:
            continue
        # Use best EMA epoch (or best regular for no-ema)
        if label != "no-ema":
            _, best_ep = get_best(rows, "val/ema_mAP_50_95", skip_ema_anomaly=(label == "no-pretrain"))
        else:
            _, best_ep = get_best(rows, "val/mAP_50_95")

        prec, rec = 0.0, 0.0
        for r in rows:
            try:
                if int(float(r["epoch"])) == best_ep:
                    prec = float(r.get("val/precision", "") or 0)
                    rec = float(r.get("val/recall", "") or 0)
                    break
            except:
                continue

        ax.scatter(rec, prec, s=150, color=color, label=display, zorder=5, edgecolors="black", linewidth=0.8)
        ax.annotate(display, (rec, prec), textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall at Best Epoch")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.2)
    ax.legend(loc="lower left")
    save_fig(fig, out_dir, "precision_recall_scatter")


def plot_delta_waterfall(data, out_dir):
    """Waterfall chart showing EMA mAP delta from baseline."""
    baseline_ema, _ = get_best(data.get("rfdetr-small-baseline", []), "val/ema_mAP_50_95")

    labels = []
    deltas = []
    colors = []
    for label, dirname, display, color in VARIANTS:
        if label == "baseline":
            continue
        rows = data.get(dirname, [])
        if not rows:
            continue
        ema_v, _ = get_best(rows, "val/ema_mAP_50_95", skip_ema_anomaly=(label == "no-pretrain"))
        if label == "no-ema":
            # Use regular mAP for no-ema since no EMA
            ema_v, _ = get_best(rows, "val/mAP_50_95")
        delta = ema_v - baseline_ema
        labels.append(display)
        deltas.append(delta)
        colors.append("#F44336" if delta < 0 else "#4CAF50")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.5)

    for bar, d in zip(bars, deltas):
        h = bar.get_height()
        y = h + 0.005 if h >= 0 else h - 0.015
        ax.text(bar.get_x() + bar.get_width()/2., y, f"{d:+.4f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=10, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ EMA mAP@50:95 vs Baseline")
    ax.set_title("EMA mAP@50:95 Delta from Baseline (0.5410)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(min(deltas) * 1.3, 0.02)

    save_fig(fig, out_dir, "delta_waterfall")


def plot_summary_table(data, out_dir):
    """Summary table as a figure."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    header = ["Variant", "Description", "Epochs", "Best mAP@50:95", "Best mAP@50", "Best EMA mAP@50:95", "Δ vs Baseline"]
    rows_data = []

    baseline_ema, _ = get_best(data.get("rfdetr-small-baseline", []), "val/ema_mAP_50_95")

    for label, dirname, display, color in VARIANTS:
        rows = data.get(dirname, [])
        if not rows:
            rows_data.append([label, display, "MISSING", "-", "-", "-", "-"])
            continue
        last_ep = max(int(float(r["epoch"])) for r in rows if r.get("epoch"))
        reg_v, reg_ep = get_best(rows, "val/mAP_50_95")
        map50_v, _ = get_best(rows, "val/mAP_50")
        ema_v, ema_ep = get_best(rows, "val/ema_mAP_50_95", skip_ema_anomaly=(label == "no-pretrain"))
        if label == "no-ema":
            ema_v = 0
        delta = ema_v - baseline_ema if ema_v > 0 else "N/A"

        rows_data.append([
            label,
            display,
            f"{last_ep+1}/30",
            f"{reg_v:.4f} (ep{reg_ep})",
            f"{map50_v:.4f}",
            f"{ema_v:.4f}" if ema_v > 0 else "N/A",
            f"{delta:+.4f}" if isinstance(delta, float) else delta,
        ])

    table = ax.table(cellText=rows_data, colLabels=header, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(header)):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight baseline row
    for j in range(len(header)):
        table[1, j].set_facecolor("#E3F2FD")

    ax.set_title("RF-DETR Ablation Study — Complete Results Summary", fontsize=14, fontweight="bold", pad=20)
    save_fig(fig, out_dir, "summary_table")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot RF-DETR ablation results")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mot/plots"))
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/mot"))
    args = parser.parse_args()

    global OUTPUT_BASE
    OUTPUT_BASE = args.data_dir

    setup_style()
    out_dir = args.output_dir

    # Load all data
    data = {}
    for label, dirname, _, _ in VARIANTS:
        rows = load_variant_csv(dirname)
        if rows:
            data[dirname] = rows
            print(f"Loaded {len(rows)} rows for {dirname}")
        else:
            print(f"WARNING: No data for {dirname}")

    print(f"\nGenerating plots in {out_dir}/ ...")

    plot_ema_map_over_epochs(data, out_dir)
    plot_reg_map_over_epochs(data, out_dir)
    plot_best_map_bar(data, out_dir)
    plot_per_class_ap(data, out_dir)
    plot_val_loss(data, out_dir)
    plot_map50_over_epochs(data, out_dir)
    plot_precision_recall_scatter(data, out_dir)
    plot_delta_waterfall(data, out_dir)
    plot_summary_table(data, out_dir)

    print(f"\nDone! All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
