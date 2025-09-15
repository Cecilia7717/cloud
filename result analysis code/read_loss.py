import re
import matplotlib.pyplot as plt
import argparse

def parse_f1_scores(log_file):
    """Extract Loss values from training logs, handling missing epochs"""
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


def plot_losses_multiple(log_files, bit):
    """Plot train and test losses from multiple log files (different seeds)."""
    plt.figure(figsize=(10, 6))

    colors = ['g', 'r', 'b']
    markers = ['o', 's', '^']

    for idx, log_file in enumerate(log_files):
        epochs, train_loss, test_loss = parse_f1_scores(log_file)
        if not epochs:
            print(f"No valid loss data found in {log_file}")
            continue

        print(f"=== Results for {log_file} ===")
        print(f"Train loss: {train_loss}")
        print(f"Test loss: {test_loss}")

        label_suffix = f"seed {idx+1}"
        plt.plot(epochs, train_loss, color=colors[idx], linestyle='-', marker=markers[idx], markersize=3,
                 label=f'train loss ({label_suffix})')
        plt.plot(epochs, test_loss, color=colors[idx], linestyle='--', marker=markers[idx], markersize=3,
                 label=f'test loss ({label_suffix})')

    plt.ylim(0.3, 0.9)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Loss by Epoch for bit = {bit}, seed = 23456', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f'loss_plot_quan_{bit}_seed_23456.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and plot loss values from training log files.")
    parser.add_argument("--bit", type=int, default=8, help="bit = 2/4/6/8")
    args = parser.parse_args()

    log_files = [
        # f"result_quan_{args.bit}_12345.txt",
        f"result_quan_{args.bit}_23456.txt"#,
        # f"result_quan_{args.bit}_34567.txt"
    ]

    plot_losses_multiple(log_files, args.bit)
