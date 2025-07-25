import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Path to your folder with CSV files
folder_path = "C:/Users/marti/Documents/Diplomka/results/train_plotting/AB_testing_max_pooling_eval_cindex"

# Use a dict to collect curves by fold number
fold_curves = defaultdict(list)

# Regex to extract fold number
fold_pattern = re.compile(r"fold[_\-]?(\d+)")

# Process each CSV file
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        match = fold_pattern.search(filename)
        if not match:
            print(f"Could not extract fold from filename: {filename}")
            continue
        fold_num = int(match.group(1))
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        if "Value" in df.columns:
            values = df["Value"].to_numpy()
            fold_curves[fold_num].append(values)
        else:
            print(f"'Value' column not found in {filename}")

# Assign a color to each fold
colors = plt.cm.get_cmap("Set1", len(fold_curves))

# Prepare mean curve across all
all_curves = [curve for fold in fold_curves.values() for curve in fold]
max_length = max(len(curve) for curve in all_curves)
curves_padded = np.array([
    np.pad(curve, (0, max_length - len(curve)), constant_values=np.nan)
    for curve in all_curves
])
mean_curve = np.nanmean(curves_padded, axis=0)

# Plotting
plt.figure(figsize=(6, 4))

for i, (fold_num, curves) in enumerate(sorted(fold_curves.items())):
    color = colors(i)
    for curve in curves:
        curve_padded = np.pad(curve, (0, max_length - len(curve)), constant_values=np.nan)
        plt.plot(curve_padded, color=color, alpha=0.4, label=f"Fold {fold_num+1}" if curve is curves[0] else "")

# Plot mean curve on top
plt.plot(mean_curve, color='black', linewidth=2.5, label="Mean Curve")

#plt.ylim(0.3, 1.0)
plt.xlabel("Epoch")
plt.ylabel("Concordance index")
plt.tick_params(axis='both', which='major', labelsize=9)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()