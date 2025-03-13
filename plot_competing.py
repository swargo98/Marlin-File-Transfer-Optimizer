import pandas as pd
import matplotlib.pyplot as plt

# File paths
file_paths = {
    "network_1": "timed_log_network_ppo_attention_write_bn_1.csv",
    "network_2": "timed_log_network_ppo_attention_write_bn_2.csv",
}

# Load all data into a dictionary
data = {}
for key, path in file_paths.items():
    try:
        data[key] = pd.read_csv(
            path, 
            header=None, 
            names=["current_time", "time_since_beginning", "throughputs", "threads"]
        )
    except FileNotFoundError:
        print(f"File not found: {path}")

# -------------------------------------------------------------------------
# Prepare DataFrames for each file:
# 1. Rename the throughput column.
# 2. Round 'current_time' to the nearest integer.
# 3. Aggregate by current_time (if multiple rows have the same time).
# -------------------------------------------------------------------------
df1 = data["network_1"].rename(columns={"threads": "tp1"})
df2 = data["network_2"].rename(columns={"threads": "tp2"})

# Keep only the necessary columns.
df1 = df1[["current_time", "tp1"]]
df2 = df2[["current_time", "tp2"]]

# Round current_time in each DataFrame before merging.
df1["current_time"] = df1["current_time"].round().astype(int)
df2["current_time"] = df2["current_time"].round().astype(int)

# In case there are duplicate current_time values, aggregate the throughput by summing.
df1 = df1.groupby("current_time", as_index=False)["tp1"].sum()
df2 = df2.groupby("current_time", as_index=False)["tp2"].sum()

# -------------------------------------------------------------------------
# Merge the two DataFrames on current_time with an outer join,
# filling missing values with 0.
# -------------------------------------------------------------------------
df_merged = pd.merge(df1, df2, on="current_time", how="outer")
df_merged.sort_values(by="current_time", inplace=True)
df_merged.fillna(0, inplace=True)

# Compute the sum of both throughputs.
df_merged["tp_sum"] = df_merged["tp1"] + df_merged["tp2"]

# -------------------------------------------------------------------------
# Apply a 5-point rolling average to each throughput column.
# -------------------------------------------------------------------------
df_merged["tp1_roll"] = df_merged["tp1"].rolling(window=5).mean()
df_merged["tp2_roll"] = df_merged["tp2"].rolling(window=5).mean()
df_merged["tp_sum_roll"] = df_merged["tp_sum"].rolling(window=5).mean()

# Save the merged DataFrame (including rolling averages) to a CSV file.
df_merged.to_csv("merged_output.csv", index=False)

# -------------------------------------------------------------------------
# Plot: rolling average throughput vs. current_time for network_1, network_2, and the sum.
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df_merged["current_time"], df_merged["tp1_roll"], label="Network 1 (5-pt rolling avg)")
plt.plot(df_merged["current_time"], df_merged["tp2_roll"], label="Network 2 (5-pt rolling avg)")
plt.plot(df_merged["current_time"], df_merged["tp_sum_roll"], label="Sum (5-pt rolling avg)")

plt.xlabel("Current Time (s)")
plt.ylabel("Throughput")
plt.title("Throughput vs. Current Time (5-pt Rolling Average)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_throughput_plot.png", dpi=300)
plt.show()