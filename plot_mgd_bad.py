import pandas as pd
import matplotlib.pyplot as plt

# File paths
file_paths = {
    "read": "logs_final/timed_log_read_ppo_mgd_network_bn.csv",
    "network": "logs_final/timed_log_network_ppo_mgd_network_bn.csv",
    "write": "logs_final/timed_log_write_ppo_mgd_network_bn.csv",
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

# Create the figure
plt.figure(figsize=(8, 5))

# Define the order in which to plot and styling
plot_order = ["read", "network", "write"]
plot_styles = {
    "read": {"linestyle": "--", "color": "red",    "label": "Read=150Mbps"},
    "network": {"linestyle": "-.", "color": "green","label": "Network=75Mbps"},
    "write": {"linestyle": ":", "color": "blue",   "label": "Write=150Mbps"},
}

# Plot each dataset with a 5-point rolling average
for key in plot_order:
    if key in data:
        concurrency_rolled = data[key]["threads"].rolling(window=5).mean()
        sample_numbers = concurrency_rolled.index
        
        plt.plot(sample_numbers, concurrency_rolled,
                 linestyle=plot_styles[key]["linestyle"],
                 color=plot_styles[key]["color"],
                 label=plot_styles[key]["label"])

# If you want to include "falcon" in the same plot:
if "falcon" in data:
    concurrency_rolled = data["falcon"]["threads"].rolling(window=5).mean()
    sample_numbers = concurrency_rolled.index
    
    plt.plot(sample_numbers, concurrency_rolled,
             linestyle=plot_styles["falcon"]["linestyle"],
             color=plot_styles["falcon"]["color"],
             label=plot_styles["falcon"]["label"])

# Set labels, legend, and layout
plt.title("Link Bandwidth = 1000Mbps")
plt.xlabel("Sample Transfer Number")
plt.ylabel("Concurrency (5-point rolling average)")
plt.legend()
plt.tight_layout()

# Save the figure to a PNG file
plt.savefig("plot_mgd_bad.png", dpi=300)  # You can adjust dpi as needed

# Display the plot
plt.show()
