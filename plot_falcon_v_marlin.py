# import pandas as pd
# import matplotlib.pyplot as plt

# # File paths
# file_paths = {
#     "falcon": "logs_final/timed_log_network_falcon_network_bn.csv"
# }

# # Load all data into a dictionary
# data = {}
# for key, path in file_paths.items():
#     try:
#         data[key] = pd.read_csv(
#             path, 
#             header=None, 
#             names=["current_time", "time_since_beginning", "throughputs", "threads"]
#         )
#     except FileNotFoundError:
#         print(f"File not found: {path}")

# # Create the figure
# plt.figure(figsize=(8, 5))

# # Define the order in which to plot and styling
# plot_order = ["read", "network", "write"]
# plot_styles = {
#     "read": {"linestyle": "--", "color": "red",    "label": "Read=150Mbps"},
#     "network": {"linestyle": "-.", "color": "green","label": "Network=75Mbps"},
#     "write": {"linestyle": ":", "color": "blue",   "label": "Write=150Mbps"},
#     "falcon": {"linestyle": "-", "color": "orange","label": "Falcon=75Mbps"}
# }

# # Plot each dataset with a 5-point rolling average
# for key in plot_order:
#     if key in data:
#         concurrency_rolled = data[key]["threads"].rolling(window=5).mean()
#         sample_numbers = concurrency_rolled.index
        
#         plt.plot(sample_numbers, concurrency_rolled,
#                  linestyle=plot_styles[key]["linestyle"],
#                  color=plot_styles[key]["color"],
#                  label=plot_styles[key]["label"])

# # If you want to include "falcon" in the same plot:
# if "falcon" in data:
#     concurrency_rolled = data["falcon"]["threads"].rolling(window=5).mean()
#     sample_numbers = concurrency_rolled.index
    
#     plt.plot(sample_numbers, concurrency_rolled,
#              linestyle=plot_styles["falcon"]["linestyle"],
#              color=plot_styles["falcon"]["color"],
#              label=plot_styles["falcon"]["label"])

# # Set labels, legend, and layout
# plt.title("Link Bandwidth = 1000Mbps")
# plt.xlabel("Sample Transfer Number")
# plt.ylabel("Concurrency (5-point rolling average)")
# plt.legend()
# plt.tight_layout()

# # Save the figure to a PNG file
# plt.savefig("plot_falcon_bad.png", dpi=300)  # You can adjust dpi as needed

# # Display the plot
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# File paths
file_paths = {
    "falcon": "logs_final/timed_log_network_falcon_network_bn.csv"
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

# -----------------------------------------------------
# Example plotting code
# -----------------------------------------------------
df = data["falcon"]

# If your throughput column is not already in Gbps, you may need to scale it.
# For instance, if 'throughputs' is in Mbps, you can convert:
# df["throughputs"] = df["throughputs"] / 1000.0  # from Mbps to Gbps

time_s = data["falcon"]["time_since_beginning"]  # or df["time_since_beginning"], whichever you prefer
throughput = data["falcon"]["throughputs"]
concurrency = data["falcon"]["threads"]

# Create the figure and the first axis (for throughput)
fig, ax1 = plt.subplots(figsize=(8, 5))

color_throughput = "red"
ax1.set_xlabel("Duration (Seconds)", fontsize=14)
ax1.set_ylabel("Throughput (Mbps)", fontsize=14)
# Plot throughput on ax1
line1 = ax1.plot(time_s, throughput, color=color_throughput, 
                 linewidth=1, label="Throughput")
ax1.tick_params(axis='y')

# Optionally set y-limits for throughput
ax1.set_ylim(0, 1050)  # for example
ax1.set_xlim(0, 150)  # for example

# Create the second axis (for concurrency)
ax2 = ax1.twinx()
color_concurrency = "green"
ax2.set_ylabel("Concurrency", fontsize=14)
line2 = ax2.plot(time_s, concurrency, color=color_concurrency, 
                 linewidth=1, linestyle="--", label="Concurrency")
ax2.tick_params(axis='y')

# Optionally set y-limits for concurrency
ax2.set_ylim(0, 60)  # for example
ax2.set_xlim(0, 150)  # for example

# Combine legend entries from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best", fontsize=12)

plt.grid(True, which='both', axis='both', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("plot_falcon_bad.pdf", dpi=300, format='pdf')
plt.show()
