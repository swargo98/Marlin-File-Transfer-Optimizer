import pandas as pd
import matplotlib.pyplot as plt

# Define types for comparison
types = ["ppo_residual_network_bn_std"]  # Add more types as needed

# Generate file paths dynamically
file_paths = {}
for t in types:
    file_paths[f"read_{t}"] = f"timed_log_read_{t}.csv"
    file_paths[f"network_{t}"] = f"timed_log_network_{t}.csv"
    file_paths[f"write_{t}"] = f"timed_log_write_{t}.csv"

# Load all data into a dictionary
data = {}
for key, path in file_paths.items():
    try:
        data[key] = pd.read_csv(path, header=None, names=["current_time", "time_since_beginning", "throughputs", "threads"])
    except FileNotFoundError:
        print(f"File not found: {path}")

median_top5 = {}
top = {}

for key, df in data.items():
    # Compute the throughput per thread ratio
    df["throughput_per_thread"] = df["throughputs"] / df["threads"]
    
    # Select the 5 largest throughput per thread values
    top5 = df["throughput_per_thread"].nlargest(101)
    
    # Compute the median of these 5 values
    median_top5[key] = top5.median()
    top[key] = df["throughputs"].nlargest(1)

# Print the median of the top 5 values for each key
print(" Highest Throughput/Thread")
for key, median_val in median_top5.items():
    print(f"{key}: {median_val}")
print(" Highest Throughput")
for key, top_val in top.items():
    print(f"{key}: {top_val}")

data = {}
top = {}
data['sender'] = pd.read_csv(f"shared_memory_log_sender_{t}.csv", header=None, names=["used_memory"])
data['receiver'] = pd.read_csv(f"shared_memory_log_receiver_{t}.csv", header=None, names=["used_memory"])

for key, df in data.items():
    top[key] = df["used_memory"].nlargest(1)

# Print the median of the top 5 values for each key
print(" Highest used memory")
for key, top_val in top.items():
    print(f"{key}: {top_val}")