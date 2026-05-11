"""
plot_comparison.py — Compare three setups:
  1. 10-node Baseline, NO straggler  (derived from baseline.csv, delay=0)
  2. 10-node Baseline, WITH straggler (baseline.csv as-is)
  3. 100-node Erasure Coded, WITH straggler (overhead_metrics.json)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "results"

# ── 1. 10-node Baseline WITHOUT straggler ────────────────────────────────────
# Same accuracy/loss as baseline.csv (straggler only adds sleep, not different
# data or different updates — baseline has no deadline so all clients finish).
# avg_delay set to 0 since ENABLE_STRAGGLER=False removes the artificial sleep.
baseline_no_straggler = {
    "label":     "10-node Baseline (no straggler)",
    "color":     "#4e9af1",
    "marker":    "o",
    "rounds":    [0, 1, 2, 3, 4, 5],
    "acc":       [0.042, 0.3968, 0.3955, 0.4651, 0.4884, 0.4975],
    "loss":      [2.3072, 1.9661, 1.6468, 1.5050, 1.4456, 1.2925],
    "train_loss":[float("nan"), 0.3078, 0.1910, 0.1522, 0.1301, 0.1248],
    "delay":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "bytes":     [0.0, 4.23, 4.23, 4.23, 4.23, 4.23],
    "clients":   10,
    "straggler": False,
}

# ── 2. 10-node Baseline WITH straggler ───────────────────────────────────────
baseline_straggler = {
    "label":     "10-node Baseline (with straggler)",
    "color":     "#e76f51",
    "marker":    "s",
    "rounds":    [0, 1, 2, 3, 4, 5],
    "acc":       [0.042, 0.3968, 0.3955, 0.4651, 0.4884, 0.4975],
    "loss":      [2.3072, 1.9661, 1.6468, 1.5050, 1.4456, 1.2925],
    "train_loss":[float("nan"), 0.3078, 0.1910, 0.1522, 0.1301, 0.1248],
    "delay":     [0.0, 1.33, 0.72, 1.24, 1.51, 1.61],
    "bytes":     [0.0, 4.23, 4.23, 4.23, 4.23, 4.23],
    "clients":   10,
    "straggler": True,
}

# ── 3. 100-node Erasure Coded WITH straggler ─────────────────────────────────
ec_100 = {
    "label":     "100-node Erasure Coded (with straggler)",
    "color":     "#2a9d8f",
    "marker":    "^",
    "rounds":    [0, 1, 2, 3, 4, 5],
    "acc":       [0.0813, 0.1000, 0.2515, 0.3269, 0.4237, 0.3649],
    "loss":      [2.3069, 2.4204, 2.5752, 2.2651, 2.3938, 1.7229],
    "train_loss":[float("nan"), 1.5149, 0.5825, 0.4859, 0.4202, 0.4007],
    "delay":     [0.0, 0.51, 0.60, 0.65, 0.60, 0.62],
    "bytes":     [0.0, 42.35, 42.35, 42.35, 42.35, 42.35],
    "clients":   100,
    "straggler": True,
}

SETUPS = [baseline_no_straggler, baseline_straggler, ec_100]


def plot_accuracy_and_loss():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Accuracy & Loss Comparison Across Setups", fontsize=13, fontweight="bold")

    # Small x-offsets so overlapping lines remain visible
    x_offsets = [-0.08, 0.08, 0.0]
    for s, xoff in zip(SETUPS, x_offsets):
        xs = [r + xoff for r in s["rounds"]]
        ax1.plot(xs, [a * 100 for a in s["acc"]],
                 marker=s["marker"], color=s["color"], label=s["label"],
                 linewidth=2, markersize=7)
        ax2.plot(xs, s["loss"],
                 marker=s["marker"], color=s["color"], label=s["label"],
                 linewidth=2, markersize=7)

    ax1.set_title("Centralized Accuracy (%)")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks([0, 1, 2, 3, 4, 5])
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 60)

    ax2.set_title("Centralized Loss")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.set_xticks([0, 1, 2, 3, 4, 5])
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "comparison_accuracy_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_delay_and_bytes():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Communication Overhead Comparison", fontsize=13, fontweight="bold")

    rounds = [1, 2, 3, 4, 5]
    for s in SETUPS:
        delays = s["delay"][1:]
        bytes_ = s["bytes"][1:]
        ax1.plot(rounds, delays, marker=s["marker"], color=s["color"],
                 label=s["label"], linewidth=2, markersize=7)
        ax2.plot(rounds, bytes_, marker=s["marker"], color=s["color"],
                 label=s["label"], linewidth=2, markersize=7)

    ax1.set_title("Avg Straggler Delay per Round (s)")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Seconds")
    ax1.set_xticks(rounds)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_title("Total Bytes Sent per Round (MB)")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("MB")
    ax2.set_xticks(rounds)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "comparison_delay_bytes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_table():
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")

    col_labels = [
        "Setup", "Nodes", "Straggler",
        "R1 Acc", "R3 Acc", "R5 Acc",
        "Avg Delay (s)", "Bytes/Round (MB)", "R5 Train Loss"
    ]

    rows = []
    for s in SETUPS:
        rows.append([
            s["label"].replace("\n", " ").strip(),
            str(s["clients"]),
            "Yes" if s["straggler"] else "No",
            f"{s['acc'][1]*100:.1f}%",
            f"{s['acc'][3]*100:.1f}%",
            f"{s['acc'][5]*100:.1f}%",
            f"{np.mean(s['delay'][1:]):.2f}s",
            f"{s['bytes'][1]:.2f}",
            f"{s['train_loss'][5]:.4f}",
        ])

    row_colors = [
        ["#e8f4fd"] * len(col_labels),
        ["#fde8e4"] * len(col_labels),
        ["#e4f5f2"] * len(col_labels),
    ]

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))
    tbl.scale(1, 2.0)

    # Bold header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Summary Comparison — 10-node vs 100-node Erasure Coded FL",
        fontsize=12, fontweight="bold", pad=30
    )

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "comparison_summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_text_table():
    print("\n" + "=" * 90)
    print(f"{'Setup':<35} {'Nodes':>6} {'Strag':>6} {'R1 Acc':>8} {'R3 Acc':>8} "
          f"{'R5 Acc':>8} {'AvgDelay':>10} {'Bytes/Rd':>10}")
    print("=" * 90)
    for s in SETUPS:
        print(
            f"{s['label'].replace(chr(10),' '):<35} "
            f"{s['clients']:>6} "
            f"{'Yes' if s['straggler'] else 'No':>6} "
            f"{s['acc'][1]*100:>7.1f}% "
            f"{s['acc'][3]*100:>7.1f}% "
            f"{s['acc'][5]*100:>7.1f}% "
            f"{np.mean(s['delay'][1:]):>9.2f}s "
            f"{s['bytes'][1]:>8.2f} MB"
        )
    print("=" * 90)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\nGenerating comparison charts...\n")
    plot_accuracy_and_loss()
    plot_delay_and_bytes()
    plot_summary_table()
    print_text_table()
    print("\nDone. All comparison charts saved to results/")
