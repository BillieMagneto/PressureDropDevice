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
    #"reference_(no_block).txt",
    "466.1.7.21_675S10-MAG118Z1.txt",
    "459-3.7.17_603S10-MAG110Z1.txt",
    "456-1.2.17_605S10-MAG112Z1.txt"
    #"454-2.1.8_585S10MAG107Z1.txt",
    #"452-2.2.44_580S10MAG102Z1.txt",
    #"448-2.1.23_557S10MAG103Z1.txt",
   # "445-0.0.26_552S10-MAG87Z1CSP10.txt",
   # "438-2.2.2_501S10-MAG99Z1.txt",

]
#"Block454-2.1.8.txt",
plt.figure(figsize=(8, 5))

for file in filenames:
    file_path = os.path.join(data_folder, file)

# Load data correctly — assign proper column names
    data = pd.read_csv(
        file_path,
        skiprows=3,  # skip metadata lines
        header=0,    # read actual header (but it’s not used)
        names=["Time", "Differential_pressure", "Flow_rate[L/m]", "Pump"]
    )

    # Filter out high flow rates
    data = data[data["Flow_rate[L/m]"] <= 5.0]

    # Clean and extract columns
    flow = data["Flow_rate[L/m]"]
    pressure_drop = data["Differential_pressure"]

    # Use the filename (without extension) as the label
    label = os.path.splitext(file)[0]
    plt.plot(flow, pressure_drop, marker='.', linestyle='', label=label)

# Labels, title, legend
plt.xlabel("Flow rate [L/min]")
plt.ylabel("Pressure drop [mbar]")
plt.title("Pressure Drop vs Flow Rate — All Measurements")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
