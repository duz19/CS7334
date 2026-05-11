"""
plot_overhead.py — Read results/overhead_metrics.json and produce:
  1. Stacked bar: per-round FL overhead breakdown
  2. System counters: memory, CPU, disk I/O over rounds
  3. Bottleneck summary table
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

METRICS_FILE = "results/overhead_metrics.json"
OUT_DIR      = "results"

PHASE_COLORS = {
    "Data Load I/O":    "#4e9af1",
    "Forward Pass":     "#f4a261",
    "Backward Pass":    "#e76f51",
    "Serialization":    "#2a9d8f",
    "Compression":      "#8ecae6",
    "Straggler Delay":  "#e9c46a",
    "Aggregation":      "#a8dadc",
    "Evaluation":       "#6a4c93",
}


def load_metrics():
    if not os.path.exists(METRICS_FILE):
        print(f"[ERROR] {METRICS_FILE} not found. Run 'flwr run . --stream' first.")
        sys.exit(1)
    with open(METRICS_FILE) as f:
        return json.load(f)


def plot_overhead_stacked(records):
    rounds = [r["round"] for r in records]
    x      = np.arange(len(rounds))
    width  = 0.55

    phases = {
        "Data Load I/O":   [r.get("avg_t_data_load_s",      0) for r in records],
        "Forward Pass":    [r.get("avg_t_forward_s",         0) for r in records],
        "Backward Pass":   [r.get("avg_t_backward_s",        0) for r in records],
        "Serialization":   [r.get("avg_t_serialization_s",   0) for r in records],
        "Compression":     [r.get("avg_t_compression_s",     0) for r in records],
        "Straggler Delay": [r.get("avg_delay_s",             0) for r in records],
        "Aggregation":     [r.get("t_aggregation_s",         0) for r in records],
        "Evaluation":      [r.get("t_evaluation_s",          0) for r in records],
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(rounds))

    for label, values in phases.items():
        vals = np.array(values)
        ax.bar(x, vals, width, bottom=bottom,
               color=PHASE_COLORS[label], label=label, edgecolor="white", linewidth=0.4)
        # Annotate non-trivial segments
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.05:
                ax.text(i, b + v / 2, f"{v:.2f}s", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r}" for r in rounds])
    ax.set_ylabel("Time (seconds)")
    ax.set_title(
        f"FL Round Overhead Breakdown — {records[0].get('mode','?')} "
        f"({records[0].get('n_clients', '?')}+ clients)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(bottom) * 1.15)

    path = os.path.join(OUT_DIR, "overhead_stacked_bar.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_system_counters(records):
    rounds   = [r["round"]               for r in records]
    mem      = [r.get("avg_mem_mb",  0)  for r in records]
    cpu      = [r.get("avg_cpu_pct", 0)  for r in records]
    disk_r   = [r.get("sum_disk_read_mb",  0) for r in records]
    disk_w   = [r.get("sum_disk_write_mb", 0) for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Low-Level Performance Counters (100-Node Simulation)",
                 fontsize=13, fontweight="bold")

    def _line(ax, y, title, ylabel, color):
        ax.plot(rounds, y, marker="o", color=color, linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(rounds)
        ax.grid(alpha=0.3)
        for rx, ry in zip(rounds, y):
            ax.annotate(f"{ry:.1f}", (rx, ry), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8)

    _line(axes[0, 0], mem,    "Avg Client Memory (RSS)", "MB",   "#4e9af1")
    _line(axes[0, 1], cpu,    "Avg Client CPU %",        "%",    "#e76f51")
    _line(axes[1, 0], disk_r, "Total Disk Read",         "MB",   "#2a9d8f")
    _line(axes[1, 1], disk_w, "Total Disk Write",        "MB",   "#f4a261")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "overhead_system_counters.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bottleneck_table(records):
    phases = ["Data Load I/O", "Forward Pass", "Backward Pass",
              "Serialization", "Compression", "Straggler Delay",
              "Aggregation", "Evaluation"]
    keys   = ["avg_t_data_load_s", "avg_t_forward_s", "avg_t_backward_s",
              "avg_t_serialization_s", "avg_t_compression_s", "avg_delay_s",
              "t_aggregation_s", "t_evaluation_s"]

    totals = []
    for k in keys:
        totals.append(np.mean([r.get(k, 0) for r in records]))

    grand   = sum(totals) or 1.0
    pcts    = [t / grand * 100 for t in totals]
    rounds_str = [str(r["round"]) for r in records]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")

    col_labels = ["Phase", "Avg Time (s)", "% of Total", "Bottleneck?"]
    rows = []
    for ph, tot, pct in zip(phases, totals, pcts):
        flag = "★ BOTTLENECK" if pct == max(pcts) else ("⚠ high" if pct > 20 else "")
        rows.append([ph, f"{tot:.3f}", f"{pct:.1f}%", flag])

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Highlight bottleneck rows
    max_pct = max(pcts)
    for i, pct in enumerate(pcts):
        color = "#ffe0e0" if pct == max_pct else ("#fff3cd" if pct > 20 else "white")
        for j in range(4):
            tbl[i + 1, j].set_facecolor(color)

    ax.set_title(
        "FL Overhead Bottleneck Summary — averaged over all rounds",
        fontsize=11, fontweight="bold", pad=20
    )

    path = os.path.join(OUT_DIR, "overhead_bottleneck_table.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = load_metrics()

    if not records:
        print("[ERROR] overhead_metrics.json is empty.")
        sys.exit(1)

    print(f"\nLoaded {len(records)} round records from {METRICS_FILE}")
    print("Generating overhead charts...\n")

    plot_overhead_stacked(records)
    plot_system_counters(records)
    plot_bottleneck_table(records)

    print("\nDone. All overhead charts saved to results/")

    # Print quick text summary
    print("\n=== BOTTLENECK ANALYSIS ===")
    keys   = ["avg_t_data_load_s", "avg_t_forward_s", "avg_t_backward_s",
              "avg_t_serialization_s", "avg_t_compression_s", "avg_delay_s",
              "t_aggregation_s", "t_evaluation_s"]
    labels = ["Data Load", "Forward", "Backward", "Serialization",
              "Compression", "Straggler Delay", "Aggregation", "Evaluation"]
    totals = [np.mean([r.get(k, 0) for r in records]) for k in keys]
    grand  = sum(totals) or 1.0
    for lb, tot in sorted(zip(labels, totals), key=lambda x: -x[1]):
        print(f"  {lb:<20} {tot:.3f}s  ({tot/grand*100:.1f}%)")


if __name__ == "__main__":
    main()
