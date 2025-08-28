import re
import matplotlib.pyplot as plt
import numpy as np

def find_best_epoch(log_file, metrics):
    """Return the epoch and value of the lowest test loss (before epoch 100)"""
    best_epoch, best_loss, best_metrics_value = None, float("inf"), float("inf")
    with open(log_file, 'r') as f:
        for line in f:
            epoch_match = re.search(r"Epoch\[(\d+)\]", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                if current_epoch >= 100:
                    break
                test = False
                for _ in range(5):
                    next_line = f.readline()
                    if not next_line:
                        break
                    if "Test" in next_line:
                        test = True
                    if test and metrics in next_line:
                        matches = re.findall(r"\d+\.\d*", next_line)
                        if len(matches) >= 2:
                            # print("correct")
                            metrics_value = float(matches[0])
                            # metrics_value = float(matches[1])
                            # metrics_value = metrics_value/2
                    if test and "Loss" in next_line:
                        # print(next_line)
                        matches = re.findall(r"\d+\.\d+", next_line)
                        if matches:
                            loss = float(matches[0])
                            if loss <= best_loss:
                                best_loss = loss
                                best_epoch = current_epoch
                                best_metrics_value = metrics_value
    return best_epoch, best_loss, best_metrics_value

if __name__ == "__main__":
    seeds = ["12345", "23456", "34567"]
    bits = [2, 4, 6, 8]
    metrics = "Recall"
    heatmap_data = np.zeros((len(seeds), len(bits)))

    for i, seed in enumerate(seeds):
        for j, bit in enumerate(bits):
            log_file = f"result_quan_{bit}_{seed}.txt"
            epoch, loss, metrics_value = find_best_epoch(log_file, metrics)
            heatmap_data[i, j] = metrics_value
            print(f"[Seed={seed}, Bit={bit}] {metrics}: {metrics_value:.6f} at epoch {epoch}")

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    plt.imshow(heatmap_data, cmap="viridis", aspect="auto")
    plt.colorbar(label=f"{metrics}")

    # Annotate heatmap
    for i in range(len(seeds)):
        for j in range(len(bits)):
            plt.text(j, i, f"{heatmap_data[i, j]:.3f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    plt.xticks(range(len(bits)), bits)
    plt.yticks(range(len(seeds)), seeds)
    plt.xlabel("Quantization Bit")
    plt.ylabel("Seed")
    plt.title(f"Best {metrics} (upper bound) per Seed & Quantization Bit")
    plt.tight_layout()
    plt.savefig(f"best_{metrics}_heatmap_seeds_up.png", dpi=300)
    plt.show()
