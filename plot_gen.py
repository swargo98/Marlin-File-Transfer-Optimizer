import pandas as pd
import matplotlib.pyplot as plt

# Define types for comparison
types = ["ppo_residual", "ppo_marlin"]  # Add more types as needed
# types = ["ppo_marlin", "ppo_mgd"]  # Add more types as needed
extension = 'write_bn_60'
# Generate file paths dynamically
file_paths = {}
for t in types:
    file_paths[f"read_{t}"] = f"timed_log_read_{t}_{extension}.csv"
    file_paths[f"network_{t}"] = f"timed_log_network_{t}_{extension}.csv"
    file_paths[f"write_{t}"] = f"timed_log_write_{t}_{extension}.csv"

# Load all data into a dictionary
data = {}
for key, path in file_paths.items():
    try:
        data[key] = pd.read_csv(path, header=None, names=["current_time", "time_since_beginning", "throughputs", "threads"])
    except FileNotFoundError:
        print(f"File not found: {path}")

# Function to create a plot with 3 subplots, using a rolling average of 5 points for the y-axis metric
def generate_plot(metric, x_axis, y_axis, ylabel, filename):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f"{ylabel} vs {x_axis.replace('_', ' ').capitalize()} ({metric.capitalize()})")

    for i, key in enumerate(["read", "network", "write"]):
        for t in types:
            df_key = f"{key}_{t}"
            if df_key in data:
                df = data[df_key]
                # Compute the rolling average of 5 points for the y_axis metric
                y_values = df[y_axis].rolling(window=10).mean()
                # Since the rolling average introduces NaNs in the first few rows, align x-axis accordingly
                x_values = df[x_axis]
                axes[i].plot(x_values, y_values, label=f"{t}")
        axes[i].set_ylabel(f"{key.capitalize()} {ylabel}")
        axes[i].legend()

    axes[-1].set_xlabel(x_axis.replace("_", " ").capitalize())
    plt.tight_layout()
    plt.show()
    fig.savefig(filename)

# Generate and save all plots
for metric in ["throughputs", "threads"]:
    for x_axis in ["current_time", "time_since_beginning"]:
        ylabel = metric.capitalize()
        filename = f"{extension}_{metric}_vs_{x_axis}.png"
        generate_plot(metric, x_axis, metric, ylabel, filename)
