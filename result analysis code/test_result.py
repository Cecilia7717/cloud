import re
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directory where your files are stored
data_dir = "/Users/chenzhuo/Documents/cloud/result txt"

# Prepare results dict
results = {2: {}, 5: {}, 10: {}}

pattern = re.compile(r"Loss:\s*([0-9.]+)")

train_noises = [2, 5, 10]
noises = [2, 5, 10]

for noise in noises:
    for train_noise in train_noises:
        filename = f"result_quan_8_gn_{train_noise}_test_noise_{noise}.txt"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Missing file: {filename}")
            continue
        
        with open(filepath, "r") as f:
            text = f.read()
        
        matches = pattern.findall(text)
        if matches:
            loss = float(matches[-1])
            results[noise][train_noise] = loss

# Convert to DataFrame: rows=Bit-width, columns=Noise
df = pd.DataFrame({noise: [results[noise].get(b, None) for b in train_noises]
                   for noise in noises}, index=train_noises)
df.index.name = "Train_noise"
df.columns.name = "Noise"

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title("Loss Heatmap")
plt.show()
