import os
from pathlib import Path
import pandas as pd

# --- Folders ---
# Adjust these two lines if your layout is different
script_dir = Path(__file__).parent
data_folder = (script_dir / ".." / "Water_Data").resolve()

# Optional: limit which files to process (e.g., only today’s run)
# files = sorted(data_folder.glob("D2025-10-07 Data-Files (*.txt)"))
files = sorted(data_folder.glob("*.txt"))

if not files:
    print(f"No .txt files found in: {data_folder}")

for txt_path in files:
    try:
        # Read the file, forcing proper column names
        df = pd.read_csv(
            txt_path,
            skiprows=3,  # skip metadata lines before the table
            header=0,    # a header line exists but we will override names anyway
            names=["Time", "Differential_pressure", "Flow_rate[L/m]", "Pump"]
        )

        # Ensure numeric types (coerce bad values to NaN)
        df["Flow_rate[L/m]"] = pd.to_numeric(df["Flow_rate[L/m]"], errors="coerce")
        df["Differential_pressure"] = pd.to_numeric(df["Differential_pressure"], errors="coerce")

        # Filter: keep rows with flow <= 5 L/min
        df_filt = df[df["Flow_rate[L/m]"] <= 5.0].copy()

        # Select and rename the two output columns
        out = df_filt[["Flow_rate[L/m]", "Differential_pressure"]].rename(
            columns={
                "Flow_rate[L/m]": "Flow_rate_L_per_min",
                "Differential_pressure": "Pressure_drop_mbar"
            }
        )

        # Write to Excel with the same base name
        xlsx_path = txt_path.with_suffix(".xlsx")
        out.to_excel(xlsx_path, index=False, sheet_name="Filtered")

        print(f"✔ Wrote {len(out)} rows -> {xlsx_path.name}")
    except Exception as e:
        print(f"✖ Failed on {txt_path.name}: {e}")
