"""
Generate CSV files and plots from all 5 FL strategy experiment results.
Run: python plot_results.py
Outputs saved to: results/
"""
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

os.makedirs("results", exist_ok=True)

# ============================================================
# Raw data from experiment runs
# ============================================================

# Structure: round -> (centralized_loss, centralized_acc, train_loss, train_acc, avg_delay_s, bytes_mb, clients_used, failures)
# NaN = round had no usable results (timeout / startup)

NaN = float("nan")

RESULTS = {
    "Baseline": {
        "total_time_s": 45.02,
        "rounds": [
            # rnd  c_loss   c_acc   tr_loss  tr_acc  delay   MB     used  fail
            (0,    2.3072,  0.0420, NaN,     NaN,    NaN,    NaN,   0,    0),
            (1,    1.9661,  0.3968, 0.3078,  0.8908, 1.33,   4.23,  10,   0),
            (2,    1.6468,  0.3955, 0.1910,  0.9288, 0.72,   4.23,  10,   0),
            (3,    1.5050,  0.4651, 0.1522,  0.9422, 1.24,   4.23,  10,   0),
            (4,    1.4456,  0.4884, 0.1301,  0.9496, 1.51,   4.23,  10,   0),
            (5,    1.2925,  0.4975, 0.1248,  0.9505, 1.61,   4.23,  10,   0),
        ],
    },
    "Partial (k=5)": {
        "total_time_s": 31.30,
        "rounds": [
            (0,    2.3067,  0.1000, NaN,     NaN,    NaN,    NaN,   0,    0),
            (1,    2.1526,  0.1291, 0.3659,  0.8782, 1.80,   2.12,  5,    0),
            (2,    2.2567,  0.2241, 0.2930,  0.8824, 0.43,   2.12,  5,    0),
            (3,    2.7542,  0.3334, 0.1887,  0.9273, 0.70,   2.12,  5,    0),
            (4,    1.9005,  0.2354, 0.1655,  0.9423, 1.66,   2.12,  5,    0),
            (5,    1.6751,  0.4188, 0.1433,  0.9479, 1.95,   2.12,  5,    0),
        ],
    },
    "Deadline (8s)": {
        "total_time_s": 32.40,
        "rounds": [
            (0,    2.3084,  0.1002, NaN,     NaN,    NaN,    NaN,   0,    0),
            (1,    2.3084,  0.1002, NaN,     NaN,    NaN,    NaN,   0,    10),  # all timed out
            (2,    2.0812,  0.4256, 0.3409,  0.8612, 0.87,   4.23,  10,   0),
            (3,    1.6157,  0.4383, 0.2062,  0.9252, 1.07,   4.23,  10,   0),
            (4,    1.3706,  0.5122, 0.1571,  0.9408, 1.59,   4.23,  10,   0),
            (5,    1.3822,  0.4429, 0.1347,  0.9476, 1.30,   4.23,  10,   0),
        ],
    },
    "Async (1.5s)": {
        "total_time_s": 10.47,
        "rounds": [
            (0,    2.3002,  0.1103, NaN,     NaN,    NaN,    NaN,   0,    0),
            (1,    2.3002,  0.1103, NaN,     NaN,    NaN,    NaN,   0,    10),
            (2,    2.3002,  0.1103, NaN,     NaN,    NaN,    NaN,   0,    10),
            (3,    2.3002,  0.1103, NaN,     NaN,    NaN,    NaN,   0,    10),
            (4,    2.3002,  0.1103, NaN,     NaN,    NaN,    NaN,   0,    10),
            (5,    2.3002,  0.1103, NaN,     NaN,    NaN,    NaN,   0,    10),
        ],
    },
    "Erasure Coded": {
        "total_time_s": 35.29,
        "rounds": [
            (0,    2.3084,  0.1000, NaN,     NaN,    NaN,    NaN,   0,    0),
            (1,    2.3084,  0.1000, NaN,     NaN,    NaN,    NaN,   0,    10),  # first-round startup
            (2,    2.3667,  0.1990, 0.3073,  0.8886, 0.49,   2.12,  5,    5),
            (3,    1.9164,  0.3520, 0.1722,  0.9366, 1.05,   4.23,  10,   0),
            (4,    1.8214,  0.4312, 0.1469,  0.9464, 1.52,   4.23,  10,   0),
            (5,    2.0224,  0.4766, 0.1217,  0.9544, 1.50,   4.23,  10,   0),
        ],
    },
}

COLS = ["round", "centralized_loss", "centralized_acc",
        "train_loss", "train_acc", "avg_delay_s", "bytes_mb",
        "clients_used", "failures"]

# ============================================================
# 1. Write per-strategy CSV files
# ============================================================
for strategy, data in RESULTS.items():
    fname = "results/" + strategy.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "") + ".csv"
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for row in data["rounds"]:
            w.writerow(row)
    print(f"Saved {fname}")

# ============================================================
# 2. Write combined summary CSV
# ============================================================
with open("results/all_strategies_combined.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["strategy"] + COLS)
    for strategy, data in RESULTS.items():
        for row in data["rounds"]:
            w.writerow([strategy] + list(row))
print("Saved results/all_strategies_combined.csv")

# ============================================================
# 3. Write final-round summary CSV
# ============================================================
summary_cols = ["strategy", "final_acc", "final_loss",
                "avg_train_acc_r5", "avg_delay_r5_s",
                "total_bytes_r5_mb", "total_time_s", "straggler_tolerance"]
straggler_info = {
    "Baseline": "None (waits for all)",
    "Partial (k=5)": "None (ignores 5 clients)",
    "Deadline (8s)": "Up to any, data lost",
    "Async (1.5s)": "Unlimited, data lost",
    "Erasure Coded": "Up to 3 (no data loss)",
}

with open("results/summary_final_round.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(summary_cols)
    for strategy, data in RESULTS.items():
        last = data["rounds"][-1]
        rnd, c_loss, c_acc, tr_loss, tr_acc, delay, mb, used, fail = last
        w.writerow([
            strategy,
            c_acc,
            round(c_loss, 4),
            round(tr_acc, 4) if not np.isnan(tr_acc) else "",
            round(delay, 2)  if not np.isnan(delay)  else "",
            mb              if not np.isnan(mb)     else "",
            data["total_time_s"],
            straggler_info[strategy],
        ])
print("Saved results/summary_final_round.csv")


# ============================================================
# Plotting setup
# ============================================================
COLORS = {
    "Baseline":       "#1f77b4",
    "Partial (k=5)":  "#ff7f0e",
    "Deadline (8s)":  "#2ca02c",
    "Async (1.5s)":   "#d62728",
    "Erasure Coded":  "#9467bd",
}
MARKERS = {
    "Baseline":       "o",
    "Partial (k=5)":  "s",
    "Deadline (8s)":  "^",
    "Async (1.5s)":   "x",
    "Erasure Coded":  "D",
}

def get_series(key_idx):
    """Extract per-round series for column index in the rounds tuple."""
    out = {}
    for strategy, data in RESULTS.items():
        rounds = [r[0] for r in data["rounds"]]
        vals   = [r[key_idx] for r in data["rounds"]]
        out[strategy] = (rounds, vals)
    return out


# ============================================================
# Plot 1 — Centralized accuracy over rounds
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
for strategy, (rounds, vals) in get_series(2).items():
    r = [r for r, v in zip(rounds, vals) if not np.isnan(v)]
    v = [v for v in vals if not np.isnan(v)]
    ax.plot(r, v, marker=MARKERS[strategy], color=COLORS[strategy],
            label=strategy, linewidth=2, markersize=7)

ax.set_xlabel("Round", fontsize=12)
ax.set_ylabel("Centralized Accuracy", fontsize=12)
ax.set_title("Centralized Test Accuracy per Round\n(FashionMNIST, 10 clients, straggler=True)", fontsize=13)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xlim(-0.2, 5.4)
ax.set_ylim(0, 0.62)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plot1_accuracy_per_round.png", dpi=150)
plt.close()
print("Saved results/plot1_accuracy_per_round.png")


# ============================================================
# Plot 2 — Centralized loss over rounds
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
for strategy, (rounds, vals) in get_series(1).items():
    r = [r for r, v in zip(rounds, vals) if not np.isnan(v) and v < 3.0]
    v = [v for v in vals if not np.isnan(v) and v < 3.0]
    ax.plot(r, v, marker=MARKERS[strategy], color=COLORS[strategy],
            label=strategy, linewidth=2, markersize=7)

ax.set_xlabel("Round", fontsize=12)
ax.set_ylabel("Centralized Loss", fontsize=12)
ax.set_title("Centralized Test Loss per Round\n(FashionMNIST, 10 clients, straggler=True)", fontsize=13)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xlim(-0.2, 5.4)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plot2_loss_per_round.png", dpi=150)
plt.close()
print("Saved results/plot2_loss_per_round.png")


# ============================================================
# Plot 3 — Final-round bar chart (accuracy, delay, bytes)
# ============================================================
strategies = list(RESULTS.keys())
final_acc   = [RESULTS[s]["rounds"][-1][2]                                     for s in strategies]
final_delay = [RESULTS[s]["rounds"][-1][5] if not np.isnan(RESULTS[s]["rounds"][-1][5]) else 0 for s in strategies]
final_bytes = [RESULTS[s]["rounds"][-1][6] if not np.isnan(RESULTS[s]["rounds"][-1][6]) else 0 for s in strategies]
total_time  = [RESULTS[s]["total_time_s"]                                       for s in strategies]

x = np.arange(len(strategies))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
bar_colors = [COLORS[s] for s in strategies]
labels_short = ["Baseline", "Partial\n(k=5)", "Deadline\n(8s)", "Async\n(1.5s)", "Erasure\nCoded"]

# Accuracy
axes[0].bar(x, final_acc, color=bar_colors, edgecolor="black", linewidth=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels_short, fontsize=9)
axes[0].set_ylabel("Accuracy"); axes[0].set_title("Final-Round\nCentralized Accuracy")
axes[0].set_ylim(0, 0.65)
for i, v in enumerate(final_acc):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

# Avg delay
axes[1].bar(x, final_delay, color=bar_colors, edgecolor="black", linewidth=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(labels_short, fontsize=9)
axes[1].set_ylabel("Avg Delay (s)"); axes[1].set_title("Avg Simulated Delay\n(Round 5)")
for i, v in enumerate(final_delay):
    axes[1].text(i, v + 0.03, f"{v:.2f}s", ha="center", fontsize=8, fontweight="bold")

# Bytes sent
axes[2].bar(x, final_bytes, color=bar_colors, edgecolor="black", linewidth=0.8)
axes[2].set_xticks(x); axes[2].set_xticklabels(labels_short, fontsize=9)
axes[2].set_ylabel("MB Sent"); axes[2].set_title("Total Bytes Sent\n(Round 5)")
for i, v in enumerate(final_bytes):
    axes[2].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

fig.suptitle("Final-Round Comparison: Accuracy / Communication Delay / Bandwidth\n(5 rounds, 10 clients, straggler=True)", fontsize=12)
plt.tight_layout()
plt.savefig("results/plot3_final_round_bars.png", dpi=150)
plt.close()
print("Saved results/plot3_final_round_bars.png")


# ============================================================
# Plot 4 — Trade-off scatter: Accuracy vs Total Time
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for i, s in enumerate(strategies):
    ax.scatter(total_time[i], final_acc[i],
               color=COLORS[s], marker=MARKERS[s], s=120,
               label=s, zorder=5, edgecolors="black", linewidths=0.7)
    ax.annotate(labels_short[i].replace("\n", " "),
                (total_time[i], final_acc[i]),
                textcoords="offset points", xytext=(6, 4), fontsize=9)

ax.set_xlabel("Total Simulation Time (s)", fontsize=12)
ax.set_ylabel("Final Centralized Accuracy", fontsize=12)
ax.set_title("Accuracy vs. Total Training Time Trade-off\n(better = top-left)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plot4_accuracy_vs_time.png", dpi=150)
plt.close()
print("Saved results/plot4_accuracy_vs_time.png")


# ============================================================
# Plot 5 — Summary table as figure
# ============================================================
table_data = []
headers = ["Strategy", "Final\nAcc", "Final\nLoss", "Avg Delay\nRd5 (s)",
           "MB/Rd5", "Total\nTime (s)", "Straggler\nTolerance"]

for s in strategies:
    last = RESULTS[s]["rounds"][-1]
    c_acc   = f"{last[2]:.4f}"
    c_loss  = f"{last[1]:.4f}"
    delay   = f"{last[5]:.2f}" if not np.isnan(last[5]) else "—"
    mb      = f"{last[6]:.2f}" if not np.isnan(last[6]) else "0"
    t_time  = f"{RESULTS[s]['total_time_s']:.1f}s"
    strag   = straggler_info[s]
    table_data.append([s, c_acc, c_loss, delay, mb, t_time, strag])

fig, ax = plt.subplots(figsize=(14, 3.5))
ax.axis("off")
tbl = ax.table(
    cellText=table_data,
    colLabels=headers,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.8)

# Header row styling
for j in range(len(headers)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Row highlighting
row_colors = ["#eaf4fb", "#ffffff", "#eaf4fb", "#ffffff", "#d5f5e3"]
for i, color in enumerate(row_colors):
    for j in range(len(headers)):
        tbl[i + 1, j].set_facecolor(color)

# Highlight EC row (last row = index 4)
for j in range(len(headers)):
    tbl[5, j].set_facecolor("#d5f5e3")
    tbl[5, j].set_text_props(fontweight="bold")

ax.set_title("FL Strategy Comparison — Summary Table\n(FashionMNIST, 10 clients, 5 rounds, straggler=True)",
             fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("results/plot5_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/plot5_summary_table.png")

print("\n=== All outputs saved to results/ ===")
print("\nFiles:")
for f in sorted(os.listdir("results")):
    size = os.path.getsize(f"results/{f}")
    print(f"  {f:45s}  {size:>8,} bytes")
