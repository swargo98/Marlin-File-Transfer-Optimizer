import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) LOAD DATA FOR THE FIRST PLOT (CONCURRENCY PLOT)
# ----------------------------------------------------------------------
file_paths_1 = {
    "read": "logs_final/timed_log_read_ppo_mgd_network_bn.csv",
    "network": "logs_final/timed_log_network_ppo_mgd_network_bn.csv",
    # "write": "logs_final/timed_log_write_ppo_mgd_network_bn.csv",
}

data_1 = {}
for key, path in file_paths_1.items():
    try:
        data_1[key] = pd.read_csv(
            path,
            header=None,
            names=["current_time", "time_since_beginning", "throughputs", "threads"]
        )
    except FileNotFoundError:
        print(f"File not found: {path}")

plot_order = ["read", "network", "write"]
plot_styles = {
    "read":    {"linestyle": "--", "color": "red",   "label": "Read=150Mbps"},
    "network": {"linestyle": "-.", "color": "green", "label": "Network=75Mbps"},
    "write":   {"linestyle": ":",  "color": "blue",  "label": "Write=150Mbps"},
}

# ----------------------------------------------------------------------
# 2) LOAD DATA FOR THE SECOND PLOT (DUAL-AXIS: FALCON)
# ----------------------------------------------------------------------
file_paths_2 = {
    "falcon": "logs_final/timed_log_network_falcon_network_bn.csv"
}

data_2 = {}
for key, path in file_paths_2.items():
    try:
        data_2[key] = pd.read_csv(
            path,
            header=None,
            names=["current_time", "time_since_beginning", "throughputs", "threads"]
        )
    except FileNotFoundError:
        print(f"File not found: {path}")

# We'll extract the Falcon data into a DataFrame for convenience
df_falcon = data_2.get("falcon", pd.DataFrame())

# ----------------------------------------------------------------------
# 3) CREATE A SINGLE FIGURE WITH TWO SUBPLOTS
#    - Left Subplot: concurrency lines
#    - Right Subplot: dual-axis (throughput & concurrency)
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(12, 5))  # Adjust figure size as needed

# -------------------------
# LEFT SUBPLOT (Concurrency)
# -------------------------
ax_left = fig.add_subplot(1, 2, 1)

# Plot concurrency lines (with 5-point rolling average)
for key in plot_order:
    if key in data_1:
        concurrency_rolled = data_1[key]["threads"].rolling(window=5).mean()
        sample_numbers = concurrency_rolled.index
        
        ax_left.plot(
            sample_numbers, concurrency_rolled,
            linestyle=plot_styles[key]["linestyle"],
            color=plot_styles[key]["color"],
            label=plot_styles[key]["label"]
        )

ax_left.set_title("Link Bandwidth = 1000Mbps\n(MGD Concurrency)")
ax_left.set_xlabel("Duration (Seconds)")
ax_left.set_ylabel("Concurrency (5-point rolling avg)")
ax_left.legend()
ax_left.grid(True)

# -------------------------
# RIGHT SUBPLOT (Dual-Axis)
# -------------------------
ax_right = fig.add_subplot(1, 2, 2)

if not df_falcon.empty:
    # Optionally, if your data is in Mbps and you want Gbps, scale accordingly
    # df_falcon["throughputs"] = df_falcon["throughputs"] / 1000.0

    time_s = df_falcon["time_since_beginning"]
    throughput = df_falcon["throughputs"]
    concurrency = df_falcon["threads"]
    
    # Plot throughput on the right subplot (left y-axis)
    ax_right.set_xlabel("Duration (Seconds)", fontsize=10)
    ax_right.set_ylabel("Throughput (Mbps)", fontsize=10, color="red")
    
    line1 = ax_right.plot(time_s, throughput, color="red", linewidth=1, label="Throughput")
    ax_right.tick_params(axis='y', labelcolor="red")
    
    # Optionally set xlim/ylim
    ax_right.set_xlim(0, 150)
    ax_right.set_ylim(0, 1050)
    
    # Create the twin y-axis for concurrency
    ax2 = ax_right.twinx()
    ax2.set_ylabel("Concurrency", fontsize=10, color="green")
    line2 = ax2.plot(time_s, concurrency, color="green", linewidth=1, linestyle="--", label="Concurrency")
    ax2.tick_params(axis='y', labelcolor="green")
    ax2.set_ylim(0, 60)
    ax2.set_xlim(0, 150)

    # Combine legend entries from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_right.legend(lines, labels, loc="best", fontsize=10)
    
    ax_right.set_title("Falcon (Dual-Axis Plot)")
    ax_right.grid(True, which='both', axis='both', linestyle=':', alpha=0.7)
else:
    ax_right.text(0.5, 0.5, "Falcon data not found", ha="center", va="center", fontsize=12)
    ax_right.set_title("Falcon (No Data)")

plt.tight_layout()

# ----------------------------------------------------------------------
# 4) SAVE & SHOW
# ----------------------------------------------------------------------
plt.savefig("combined_two_subplots.pdf", dpi=300, format='pdf')
plt.show()