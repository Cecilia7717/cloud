import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Directory containing result files
data_dir = "/Users/chenzhuo/Documents/cloud/result txt/test"

# Parameters
quans = [2, 4, 6, 8]
noises = [1, 2, 5]

# Regex pattern to extract "Loss: number"
pattern = re.compile(r"Loss:\s*([0-9.]+)")

# Initialize results matrix (rows=noises, cols=quans)
results = np.full((len(noises), len(quans)), np.nan)

# Collect data
for i, noise in enumerate(noises):
    for j, quan in enumerate(quans):
        filename = f"result_test_{quan}_{noise}_sap.txt"
        if noise == 0:
            filename = f"result_test_{quan}_{noise}.txt"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"Missing file: {filename}")
            continue

        with open(filepath, "r") as f:
            text = f.read()

        matches = pattern.findall(text)
        if matches:
            loss = float(matches[-1])  # take the last loss
            results[i, j] = loss
        else:
            print(f"No loss found in: {filename}")

# Plot heatmap
plt.figure(figsize=(8, 6))
im = plt.imshow(results, cmap="viridis", aspect="auto", 
                interpolation="nearest", origin="upper")

# Axis labels
plt.xticks(range(len(quans)), quans)
plt.yticks(range(len(noises)), noises)
plt.xlabel("Quantization Bit Width")
plt.ylabel("sap Noise Level")
plt.title("Loss Heatmap")

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label("Loss")

# Annotate cells
for i in range(len(noises)):
    for j in range(len(quans)):
        if not np.isnan(results[i, j]):
            plt.text(j, i, f"{results[i, j]:.7f}",
                     ha="center", va="center", color="w")

plt.tight_layout()
plt.savefig("/Users/chenzhuo/Documents/cloud/result plot/heatmap loss noisefree trained sap", dpi=300)
plt.show()