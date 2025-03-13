# import pandas as pd
# import matplotlib.pyplot as plt

# # Define types for comparison
# types = ["ppo_residual_network_bn_std"]  # Add more types as needed

# # Generate file paths dynamically
# file_paths = {}
# for t in types:
#     file_paths[f"read_{t}"] = f"timed_log_read_{t}.csv"
#     file_paths[f"network_{t}"] = f"timed_log_network_{t}.csv"
#     file_paths[f"write_{t}"] = f"timed_log_write_{t}.csv"

# # Load all data into a dictionary
# data = {}
# for key, path in file_paths.items():
#     try:
#         data[key] = pd.read_csv(path, header=None, names=["current_time", "time_since_beginning", "throughputs", "threads"])
#     except FileNotFoundError:
#         print(f"File not found: {path}")

# median_top5 = {}
# top = {}

# for key, df in data.items():
#     # Compute the throughput per thread ratio
#     df["throughput_per_thread"] = df["throughputs"] / df["threads"]
    
#     # Select the 5 largest throughput per thread values
#     top5 = df["throughput_per_thread"].nlargest(101)
    
#     # Compute the median of these 5 values
#     median_top5[key] = top5.median()
#     top[key] = df["throughputs"].nlargest(1)

# # Print the median of the top 5 values for each key
# print(" Highest Throughput/Thread")
# for key, median_val in median_top5.items():
#     print(f"{key}: {median_val}")
# print(" Highest Throughput")
# for key, top_val in top.items():
#     print(f"{key}: {top_val}")

# data = {}
# top = {}
# data['sender'] = pd.read_csv(f"shared_memory_log_sender_{t}.csv", header=None, names=["used_memory"])
# data['receiver'] = pd.read_csv(f"shared_memory_log_receiver_{t}.csv", header=None, names=["used_memory"])

# for key, df in data.items():
#     top[key] = df["used_memory"].nlargest(1)

# # Print the median of the top 5 values for each key
# print(" Highest used memory")
# for key, top_val in top.items():
#     print(f"{key}: {top_val}")

# import pandas as pd
# import pprint

# # -------------------------------------------------
# # 1. Define model types and bottlenecks
# # -------------------------------------------------
# types = ["ppo_residual", "ppo_marlin"]  # Two model types
# bottlenecks = ["read_bn", "network_bn", "write_bn"]

# # -------------------------------------------------
# # 2. Generate file paths dynamically
# #    There are 6 experiments: each combination of model and bottleneck,
# #    with three files: read, network, and write.
# # -------------------------------------------------
# file_paths = {}
# for t in types:
#     for b in bottlenecks:
#         file_paths[f"read_{t}_{b}"]    = f"logs_final_2/timed_log_read_{t}_{b}.csv"
#         file_paths[f"network_{t}_{b}"] = f"logs_final_2/timed_log_network_{t}_{b}.csv"
#         file_paths[f"write_{t}_{b}"]   = f"logs_final_2/timed_log_write_{t}_{b}.csv"

# # -------------------------------------------------
# # 3. Load CSV data into a dictionary
# # -------------------------------------------------
# data = {}
# for key, path in file_paths.items():
#     try:
#         df = pd.read_csv(
#             path,
#             header=None,
#             names=["current_time", "time_since_beginning", "throughputs", "threads"]
#         )
#         data[key] = df
#     except FileNotFoundError:
#         print(f"File not found: {path}")

# # -------------------------------------------------
# # 4. Compute statistics for each experiment
# # -------------------------------------------------
# # We'll store stats in a dictionary where the key is "model_bottleneck" (e.g., "ppo_residual_read_bn")
# # and the value is another dict with keys "read", "network", "write" holding their respective stats.
# stats = {}

# # Loop over each experiment (model x bottleneck)
# for model in types:
#     for b in bottlenecks:
#         exp_key = f"{model}_{b}"
#         stats[exp_key] = {}
        
#         # For each file type in the experiment: "read", "network", "write"
#         for file_type in ["read", "network", "write"]:
#             key = f"{file_type}_{model}_{b}"
#             if key in data:
#                 df = data[key]
                
#                 # (1) Average and standard deviation of threads
#                 avg_threads = df["threads"].mean()
#                 std_threads = df["threads"].std()
                
#                 # (2) First time reaching 95% of maximum throughput.
#                 # Use the "time_since_beginning" column.
#                 max_tp = df["throughputs"].max()
#                 threshold = 0.95 * max_tp
#                 # Find the first occurrence where throughput >= threshold:
#                 first_reach = df[df["throughputs"] >= threshold]
#                 if not first_reach.empty:
#                     first_time_95 = first_reach.iloc[0]["time_since_beginning"]
#                 else:
#                     first_time_95 = None
                
#                 # (3) Total time to complete the experiment:
#                 # We'll use the range of "time_since_beginning"
#                 total_time = df["time_since_beginning"].max() - df["time_since_beginning"].min()
                
#                 # Store stats for this file type
#                 stats[exp_key][file_type] = {
#                     "avg_threads": avg_threads,
#                     "std_threads": std_threads,
#                     "first_time_95": first_time_95,
#                     "total_time": total_time
#                 }
#             else:
#                 stats[exp_key][file_type] = None

# # Pretty-print the computed statistics:
# pprint.pprint(stats)

# import pandas as pd
# import pprint

# # --------------------------
# # (Re)compute stats as before:
# # --------------------------
# types = ["ppo_residual", "ppo_marlin"]  # Two model types
# bottlenecks = ["read_bn", "network_bn", "write_bn"]

# # Generate file paths dynamically for 6 experiments (each combination of model and bottleneck)
# file_paths = {}
# for t in types:
#     for b in bottlenecks:
#         file_paths[f"read_{t}_{b}"]    = f"logs_final_2/timed_log_read_{t}_{b}.csv"
#         file_paths[f"network_{t}_{b}"] = f"logs_final_2/timed_log_network_{t}_{b}.csv"
#         file_paths[f"write_{t}_{b}"]   = f"logs_final_2/timed_log_write_{t}_{b}.csv"

# # Load CSV data into a dictionary
# data = {}
# for key, path in file_paths.items():
#     try:
#         df = pd.read_csv(
#             path, 
#             header=None, 
#             names=["current_time", "time_since_beginning", "throughputs", "threads"]
#         )
#         data[key] = df
#     except FileNotFoundError:
#         print(f"File not found: {path}")

# # Compute statistics for each experiment.
# stats = {}

# for model in types:
#     for b in bottlenecks:
#         exp_key = f"{model}_{b}"
#         stats[exp_key] = {}
#         for file_type in ["read", "network", "write"]:
#             key = f"{file_type}_{model}_{b}"
#             if key in data:
#                 df = data[key]
#                 # (1) Average and standard deviation of threads
#                 avg_threads = df["threads"].mean()
#                 std_threads = df["threads"].std()
                
#                 # (2) First time reaching 95% of maximum throughput (using "time_since_beginning")
#                 max_tp = df["throughputs"].max()
#                 threshold = 0.95 * max_tp
#                 first_reach = df[df["throughputs"] >= threshold]
#                 first_time_95 = first_reach.iloc[0]["time_since_beginning"] if not first_reach.empty else None
                
#                 # (3) Total experiment time (difference between max and min time_since_beginning)
#                 total_time = df["time_since_beginning"].max() - df["time_since_beginning"].min()
                
#                 stats[exp_key][file_type] = {
#                     "avg_threads": avg_threads,
#                     "std_threads": std_threads,
#                     "first_time_95": first_time_95,
#                     "total_time": total_time
#                 }
#             else:
#                 stats[exp_key][file_type] = None

# # -------------------------------------
# # Compare Marlin vs. PPO_Residual
# # We assume that the "marlin" experiments are under "ppo_marlin"
# # and the PPO_Residual experiments are under "ppo_residual".
# # -------------------------------------
# comparisons = {}

# for b in bottlenecks:
#     comparisons[b] = {}
#     for file_type in ["read", "network", "write"]:
#         key_residual = f"ppo_residual_{b}"
#         key_marlin   = f"ppo_marlin_{b}"
#         stat_residual = stats.get(key_residual, {}).get(file_type)
#         stat_marlin   = stats.get(key_marlin, {}).get(file_type)
#         if stat_residual is not None and stat_marlin is not None:
#             conv_diff = stat_marlin["first_time_95"] - stat_residual["first_time_95"] \
#                 if (stat_marlin["first_time_95"] is not None and stat_residual["first_time_95"] is not None) else None
#             avg_diff = stat_marlin["avg_threads"] - stat_residual["avg_threads"]
#             std_diff = stat_marlin["std_threads"] - stat_residual["std_threads"]
#             time_diff = stat_marlin["total_time"] - stat_residual["total_time"]
#             comparisons[b][file_type] = {
#                 "convergence_time_diff": conv_diff,   # positive means residual converges faster
#                 "avg_threads_diff": avg_diff,          # positive means Marlin uses more threads on average
#                 "std_threads_diff": std_diff,          # positive means Marlin has higher thread variability
#                 "total_time_diff": time_diff           # positive means residual experiment finished faster
#             }
#         else:
#             comparisons[b][file_type] = None

# # Pretty-print the comparisons
# print("Comparisons (Marlin minus PPO_Residual):")
# pprint.pprint(comparisons)

# import pandas as pd
# import pprint

# # -------------------------------------------------
# # 1. Define model types and bottlenecks
# # -------------------------------------------------
# types = ["ppo_residual", "ppo_marlin"]  # Two model types
# bottlenecks = ["read_bn", "network_bn", "write_bn"]

# # -------------------------------------------------
# # 2. Generate file paths dynamically
# #    There are 6 experiments: each combination of model and bottleneck,
# #    with three files: read, network, and write.
# # -------------------------------------------------
# file_paths = {}
# for t in types:
#     for b in bottlenecks:
#         file_paths[f"read_{t}_{b}"]    = f"logs_final_2/timed_log_read_{t}_{b}.csv"
#         file_paths[f"network_{t}_{b}"] = f"logs_final_2/timed_log_network_{t}_{b}.csv"
#         file_paths[f"write_{t}_{b}"]   = f"logs_final_2/timed_log_write_{t}_{b}.csv"

# # -------------------------------------------------
# # 3. Load CSV data into a dictionary
# # -------------------------------------------------
# data = {}
# for key, path in file_paths.items():
#     try:
#         df = pd.read_csv(
#             path,
#             header=None,
#             names=["current_time", "time_since_beginning", "throughputs", "threads"]
#         )
#         data[key] = df
#     except FileNotFoundError:
#         print(f"File not found: {path}")

# # -------------------------------------------------
# # 4. Compute statistics for each experiment
# # -------------------------------------------------
# stats = {}

# for model in types:
#     for b in bottlenecks:
#         exp_key = f"{model}_{b}"
#         stats[exp_key] = {}
#         for file_type in ["read", "network", "write"]:
#             key = f"{file_type}_{model}_{b}"
#             if key in data:
#                 df = data[key]
                
#                 # (1) Average and standard deviation of threads
#                 avg_threads = df["threads"].mean()
#                 std_threads = df["threads"].std()
                
#                 # (2) First time reaching 95% of maximum throughput
#                 max_tp = df["throughputs"].max()
#                 threshold = 0.95 * max_tp
#                 first_reach = df[df["throughputs"] >= threshold]
#                 first_time_95 = first_reach.iloc[0]["time_since_beginning"] if not first_reach.empty else None
                
#                 # (3) Total experiment time (difference between max and min time_since_beginning)
#                 total_time = df["time_since_beginning"].max() - df["time_since_beginning"].min()
                
#                 stats[exp_key][file_type] = {
#                     "avg_threads": avg_threads,
#                     "std_threads": std_threads,
#                     "first_time_95": first_time_95,
#                     "total_time": total_time
#                 }
#             else:
#                 stats[exp_key][file_type] = None

# # -------------------------------------
# # 5. Compare Marlin vs. PPO_Residual with percentage differences
# #    This time, percentage differences are computed relative to Marlin (ppo_marlin)
# # -------------------------------------
# comparisons = {}

# for b in bottlenecks:
#     comparisons[b] = {}
#     for file_type in ["read", "network", "write"]:
#         key_residual = f"ppo_residual_{b}"
#         key_marlin   = f"ppo_marlin_{b}"
#         stat_residual = stats.get(key_residual, {}).get(file_type)
#         stat_marlin   = stats.get(key_marlin, {}).get(file_type)
        
#         if stat_residual is not None and stat_marlin is not None:
#             # For convergence time: compare PPO_Residual relative to Marlin.
#             if stat_marlin["first_time_95"] is not None and stat_marlin["first_time_95"] != 0:
#                 conv_pct = ((stat_residual["first_time_95"] - stat_marlin["first_time_95"]) /
#                             stat_marlin["first_time_95"]) * 100
#             else:
#                 conv_pct = None
                
#             # For average threads:
#             if stat_marlin["avg_threads"] != 0:
#                 avg_pct = ((stat_residual["avg_threads"] - stat_marlin["avg_threads"]) /
#                            stat_marlin["avg_threads"]) * 100
#             else:
#                 avg_pct = None
            
#             # For standard deviation:
#             if stat_marlin["std_threads"] != 0:
#                 std_pct = ((stat_residual["std_threads"] - stat_marlin["std_threads"]) /
#                            stat_marlin["std_threads"]) * 100
#             else:
#                 std_pct = None
            
#             # For total experiment time:
#             if stat_marlin["total_time"] != 0:
#                 time_pct = ((stat_residual["total_time"] - stat_marlin["total_time"]) /
#                             stat_marlin["total_time"]) * 100
#             else:
#                 time_pct = None
            
#             comparisons[b][file_type] = {
#                 "convergence_pct": conv_pct,   # Positive means PPO_Residual takes longer than Marlin to reach 95%
#                 "avg_threads_pct": avg_pct,      # Positive means PPO_Residual has higher avg thread count than Marlin
#                 "std_threads_pct": std_pct,      # Positive means PPO_Residual has higher thread variability than Marlin
#                 "total_time_pct": time_pct       # Positive means PPO_Residual takes longer overall than Marlin
#             }
#         else:
#             comparisons[b][file_type] = None

# print("Percentage Comparisons (PPO_Residual relative to Marlin):")
# pprint.pprint(comparisons)


import pandas as pd
import pprint

# -------------------------------------------------
# 1. Define model types and bottlenecks
# -------------------------------------------------
types = ["ppo_residual", "ppo_marlin"]  # Two model types
bottlenecks = ["read_bn", "network_bn", "write_bn"]

# -------------------------------------------------
# 2. Generate file paths dynamically
#    There are 6 experiments: each combination of model and bottleneck,
#    with three files: read, network, and write.
# -------------------------------------------------
file_paths = {}
for t in types:
    for b in bottlenecks:
        file_paths[f"read_{t}_{b}"]    = f"logs_final_2/timed_log_read_{t}_{b}.csv"
        file_paths[f"network_{t}_{b}"] = f"logs_final_2/timed_log_network_{t}_{b}.csv"
        file_paths[f"write_{t}_{b}"]   = f"logs_final_2/timed_log_write_{t}_{b}.csv"

# -------------------------------------------------
# 3. Load CSV data into a dictionary
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
# 4. Compute statistics for each experiment
# -------------------------------------------------
# We'll store stats in a dictionary where the key is "model_bottleneck"
# (e.g., "ppo_residual_read_bn") and the value is another dict with keys
# "read", "network", "write" holding their respective stats.
stats = {}

# Loop over each experiment (model x bottleneck)
for model in types:
    for b in bottlenecks:
        exp_key = f"{model}_{b}"
        stats[exp_key] = {}
        
        # For each file type in the experiment: "read", "network", "write"
        for file_type in ["read", "network", "write"]:
            key = f"{file_type}_{model}_{b}"
            if key in data:
                df = data[key]
                
                # (1) Average and standard deviation of threads
                avg_threads = df["threads"].mean()
                std_threads = df["threads"].std()
                
                # (2) First time reaching 95% of maximum throughput.
                # Use the "time_since_beginning" column.
                max_tp = df["throughputs"].max()
                threshold = 0.90 * max_tp
                first_reach = df[df["throughputs"] >= threshold]
                first_time_95 = first_reach.iloc[0]["time_since_beginning"] if not first_reach.empty else None
                
                # (3) Total time to complete the experiment:
                total_time = df["time_since_beginning"].max() - df["time_since_beginning"].min()
                
                # Store stats for this file type
                stats[exp_key][file_type] = {
                    "avg_threads": avg_threads,
                    "std_threads": std_threads,
                    "first_time_95": first_time_95,
                    "total_time": total_time
                }
            else:
                stats[exp_key][file_type] = None

# Pretty-print the computed statistics by experiment:
print("Per-experiment stats:")
pprint.pprint(stats)

# -------------------------------------------------
# 5. Group statistics by stat type rather than experiment/file
# -------------------------------------------------
# We want to create a structure like:
# {
#   "avg_threads": { "read": { exp_key: value, ... }, "network": {...}, "write": {...} },
#   "std_threads": { ... },
#   "first_time_95": { ... },
#   "total_time": { ... }
# }
grouped_stats = {}
for stat in ["avg_threads", "std_threads", "first_time_95", "total_time"]:
    grouped_stats[stat] = {}
    for file_type in ["read", "network", "write"]:
        grouped_stats[stat][file_type] = {}
        # Loop over all experiments
        for exp_key, exp_stats in stats.items():
            # If there is a valid stat for this file type, record it.
            if exp_stats[file_type] is not None:
                grouped_stats[stat][file_type][exp_key] = exp_stats[file_type][stat]
            else:
                grouped_stats[stat][file_type][exp_key] = None

# Pretty-print the stats grouped by type:
print("\nStats grouped by type:")
pprint.pprint(grouped_stats)
