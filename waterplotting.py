import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Folder setup
script_dir = os.path.dirname(__file__)
data_folder = os.path.join(script_dir, "..", "Water_Data")

# List of all files to plot (edit names if needed)
filenames = [
    "Block438-2.2.2.txt",
    "Block445-0.2.26.txt",
    "Block448-2.1.23.txt",
    "Block452-2.2.44.txt",
    "Block456-1.2.17.txt",
    "Block459-3.7.17.txt",
    "Block466-1.7.21.txt"
]
#"Block454-2.1.8.txt",
plt.figure(figsize=(8, 5))

for file in filenames:
    file_path = os.path.join(data_folder, file)

    # Load data, skipping first 3 header lines
    data = pd.read_csv(file_path, skiprows=3)

    flow = data["Flow_rate[L/m]"]
    pressure_drop = data["Differential_pressure"]

    # Use the filename (without extension) as the label
    label = os.path.splitext(file)[0]
    plt.plot(flow, pressure_drop, marker='o', linestyle='', label=label)

# Labels, title, legend
plt.xlabel("Flow rate [L/min]")
plt.ylabel("Pressure drop [mbar]")
plt.title("Pressure Drop vs Flow Rate — All Measurements")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
