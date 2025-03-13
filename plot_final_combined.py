import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Define model types and bottlenecks
# -------------------------------------------------
types = ["ppo_residual", "ppo_marlin"]  # Two model types
bottlenecks = ["read_bn", "network_bn", "write_bn"]

# Subplot titles for bottlenecks
bottleneck_labels = {
    "read_bn":    "Read I/O Bottleneck",
    "network_bn": "Network Bottleneck",
    "write_bn":   "Write I/O Bottleneck"
}

# Model labels for legend and y-axis labeling
type_labels = {
    "ppo_residual": "AutoMDT",
    "ppo_marlin":    "Marlin",
}

# -------------------------------------------------
# 2. Define link speeds (throughput per thread) for each bottleneck
# -------------------------------------------------
link_speeds = {
    "network_bn": {"read": 205, "network": 75,  "write": 195},
    "read_bn":    {"read": 80,  "network": 160, "write": 200},
    "write_bn":   {"read": 200, "network": 150, "write": 70},
}

# -------------------------------------------------
# 3. Generate file paths dynamically
#    Code loads all CSVs: read, network, and write files for each type and bottleneck.
# -------------------------------------------------
file_paths = {}
for t in types:
    for b in bottlenecks:
        file_paths[f"read_{t}_{b}"]    = f"logs_final_2/timed_log_read_{t}_{b}.csv"
        file_paths[f"network_{t}_{b}"] = f"logs_final_2/timed_log_network_{t}_{b}.csv"
        file_paths[f"write_{t}_{b}"]   = f"logs_final_2/timed_log_write_{t}_{b}.csv"

# -------------------------------------------------
# 4. Load CSV data into a dictionary
# -------------------------------------------------
data = {}
for key, path in file_paths.items():
    try:
        df = pd.read_csv(
            path, 
            header=None, 
            names=["current_time", "time_since_beginning", "throughputs", "threads"]
        )
        data[key] = df
    except FileNotFoundError:
        print(f"File not found: {path}")

# -------------------------------------------------
# 5. Create a figure with 9 subplots (3 rows x 3 columns)
#    - Rows 0 and 1: Concurrency plots (threads) for each model type (AutoMDT, Marlin)
#    - Row 2: Network throughput plots (throughputs) for both models.
# -------------------------------------------------
fig, axes = plt.subplots(nrows=3, ncols=len(bottlenecks), figsize=(15, 9), sharex=False, sharey=False)

# Define common line styles for the first two rows (for read, network, write)
line_styles = {
    "read":    {"linestyle": "--", "color": "red"},
    "network": {"linestyle": "-.", "color": "green"},
    "write":   {"linestyle": ":",  "color": "blue"},
}

# -------------------------------------------------
# 6. Fill in the top two rows (concurrency plots using "threads")
# -------------------------------------------------
# Rows 0 and 1 correspond to each model type (in order of the list 'types')
for row_idx, t in enumerate(types):
    for col_idx, b in enumerate(bottlenecks):
        ax = axes[row_idx, col_idx]
        # For each subplot, plot three lines: read, network, write
        for sub_key in ["read", "network", "write"]:
            dict_key = f"{sub_key}_{t}_{b}"  # e.g. "read_ppo_marlin_read_bn"
            if dict_key in data:
                # Compute 5-point rolling average for the "threads" column
                conv_roll = data[dict_key]["threads"].rolling(window=5).mean()
                speeds_for_bottleneck = link_speeds[b]
                label_str = f"{sub_key.capitalize()} = {speeds_for_bottleneck[sub_key]} Mbps"
                ax.plot(conv_roll.index, conv_roll, label=label_str, **line_styles.get(sub_key, {}))
        
        # Set column title on the top row only
        if row_idx == 0:
            ax.set_title(f"{bottleneck_labels[b]}\n")
        
        # Set y-axis labels: include the model type for each subplot.
        ax.set_ylabel(f"Concurrency ({type_labels.get(t, t)})")
        
        ax.set_xlabel("Duration (Seconds)")

        # ax.set_xlim(0, 270)
        # ax.set_xticks(range(0, 270, 50))
        ax.set_ylim(0, 21)
        ax.set_yticks(range(0, 21, 5))

        ax.grid(True)
        ax.legend(fontsize=8)

# -------------------------------------------------
# 7. Fill in the bottom row (row index 2): network throughput plots
#    Here we plot the "throughputs" column (from network CSV) for each model type.
# -------------------------------------------------
for col_idx, b in enumerate(bottlenecks):
    ax = axes[2, col_idx]
    for t in types:
        key = f"network_{t}_{b}"  # e.g. "network_ppo_marlin_network_bn"
        if key in data:
            # Compute 5-point rolling average for the "throughputs" column
            tp_roll = data[key]["throughputs"].rolling(window=5).mean()
            label = f"{type_labels[t]}"
            ax.plot(tp_roll.index, tp_roll, label=label)
    
    ax.set_xlabel("Duration (Seconds)")
    ax.set_ylabel("Throughput (Mbps)")
    
    # ax.set_xlim(0, 270)
    # ax.set_xticks(range(0, 270, 50))    
    ax.set_ylim(0, 1050)
    ax.set_yticks(range(0, 1050, 200))
    
    ax.grid(True)
    ax.legend(fontsize=8)

# -------------------------------------------------
# 8. For each column, set xlim to be (0, max_xlim) for all subplots in that column
# -------------------------------------------------
nrows, ncols = axes.shape
for col in range(ncols):
    max_xlim = 0
    # Find maximum x-limit (right-hand value) for this column
    for row in range(nrows):
        xlim = axes[row, col].get_xlim()
        if xlim[1] > max_xlim:
            max_xlim = xlim[1]
    # Set xlim for every subplot in this column to (0, max_xlim)
    for row in range(nrows):
        axes[row, col].set_xlim(0, max_xlim)
        axes[row, col].set_xticks(range(0, int(max_xlim), 50))    

plt.tight_layout()
plt.savefig("combined_9subplots.pdf", dpi=300, format='pdf')
plt.show()
