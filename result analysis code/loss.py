import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_f1_scores(log_file):
    """Extract training and testing Loss values from training logs, handling missing epochs"""
    epochs = []
    train_loss = []
    test_loss = []

    with open(log_file, 'r') as f:
        for line in f:
            epoch_match = re.search(r"Epoch\[(\d+)\]", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                if current_epoch >= 100:
                    break
                test = False
                # look ahead a few lines for losses
                for _ in range(5):
                    next_line = f.readline()
                    if not next_line:
                        break
                    if "Test" in next_line:
                        test = True
                    if "Loss" in next_line:
                        matches = re.findall(r"\d+\.\d+", next_line)
                        if len(matches) >= 1:
                            if test:
                                epochs.append(current_epoch)
                                test_loss.append(float(matches[0]))
                            else:
                                train_loss.append(float(matches[0]))
                            break
    return epochs, train_loss, test_loss


def main():
    base_dir = Path("/Users/chenzhuo/Documents/cloud/result txt")
    files = {
        "2-bit": base_dir / "quan_2_new.txt",
        "4-bit": base_dir / "quan_4_new.txt",
        "6-bit": base_dir / "quan_6_new.txt",
        "8-bit": base_dir / "quan_8_new.txt",
    }

    # Assign distinct colors for each bit-width
    colors = {
        "2-bit": "tab:red",
        "4-bit": "tab:blue",
        "6-bit": "tab:green",
        "8-bit": "tab:orange",
    }

    plt.figure(figsize=(10, 6))

    for label, path in files.items():
        if not path.exists():
            print(f"Missing file: {path}")
            continue

        epochs, train_loss, test_loss = parse_f1_scores(path)

        # Align lengths in case of missing data
        n = min(len(train_loss), len(test_loss))
        epochs = epochs[:n]

        color = colors.get(label, None)

        plt.plot(
            epochs[:len(train_loss)],
            train_loss,
            label=f"{label} Train Loss",
            linestyle="-",
            color=color,
        )
        plt.plot(
            epochs[:len(test_loss)],
            test_loss,
            label=f"{label} Test Loss",
            linestyle="--",
            color=color,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Testing Loss across Quantization Bit-widths")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/Users/chenzhuo/Documents/cloud/result plot/loss plot with no onnx model", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
