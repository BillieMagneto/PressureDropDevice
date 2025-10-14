import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Folder setup
script_dir = os.path.dirname(__file__)
data_folder = os.path.join(script_dir, "..", "data")

# List your 8 data files
filenames = [
    "D2025-10-07 Data-Files (1).txt",
    "D2025-10-07 Data-Files (2).txt",
    "D2025-10-07 Data-Files (3).txt",
    "D2025-10-07 Data-Files (4).txt",
    "D2025-10-07 Data-Files (5).txt",
    "D2025-10-07 Data-Files (6).txt",
    "D2025-10-07 Data-Files (7).txt",
    "D2025-10-07 Data-Files (8).txt"
]

plt.figure(figsize=(8, 5))

for file in filenames:
    file_path = os.path.join(data_folder, file)
    data = pd.read_csv(file_path, skiprows=3)

    # Extract columns
    x = data["Flow_rate[L/m]"].values.reshape(-1, 1)
    y = data["Differential_pressure"].values

    # Linear regression
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # Calculate R²
    r2 = r2_score(y, y_pred)

    # Plot data points
    plt.scatter(x, y, label=f"{file[:-4]} data", alpha=0.6)

    # Plot fitted line
    plt.plot(x, y_pred, linestyle='--', label=f"{file[:-4]} fit (R²={r2:.3f})")

    # Print stats to console
    print(f"{file}: slope = {model.coef_[0]:.4f}, intercept = {model.intercept_:.4f}, R² = {r2:.4f}")

# Labels and formatting
plt.xlabel("Flow rate [L/min]")
plt.ylabel("Pressure drop [mbar]")
plt.title("Pressure Drop vs Flow Rate — Linear Fits")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
