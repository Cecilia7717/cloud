import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_f1_scores(log_file, metrics):
    """Extract F1 scores from training logs, handling missing epochs"""
    epochs = []
    f1_1 = []
    f1_2 = []
    f_test_1 = []
    f_test_2 = []

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
                    if metrics in next_line:
                        matches = re.findall(r"\d+\.\d+", next_line)
                        if len(matches) >= 2:
                            if test:
                                epochs.append(current_epoch)
                                f_test_1.append(float(matches[0]))
                                f_test_2.append(float(matches[1]))
                            else:
                                f1_1.append(float(matches[0]))
                                f1_2.append(float(matches[1]))
                            break
    return epochs, f1_1, f1_2, f_test_1, f_test_2


def plot_f1_scores_multiple(log_files, metrics, bit):
    """Plot all F1 score lines from multiple log files in one plot."""
    plt.figure(figsize=(10, 6))

    colors = ['g', 'r', 'b']
    markers = ['o', 's', '^']

    for idx, log_file in enumerate(log_files):
        epochs, f1_1, f1_2, f_test_1, f_test_2 = parse_f1_scores(log_file, metrics)
        if not epochs:
            print(f"No valid F1 data found in {log_file}")
            continue

        print(f"=== Results for {log_file} ===")
        print(f"Class 1 train: {f1_1}")
        print(f"Class 2 train: {f1_2}")
        print(f"Class 1 test: {f_test_1}")
        print(f"Class 2 test: {f_test_2}")

        label_suffix = f"seed {idx+1}"
        plt.plot(epochs, f1_1, color=colors[idx], linestyle='-', marker=markers[idx], markersize=3,
                 label=f'train 1 ({label_suffix})')
        plt.plot(epochs, f1_2, color=colors[idx], linestyle='-', marker=markers[idx], markersize=3,
                 label=f'train 2 ({label_suffix})')
        plt.plot(epochs, f_test_1, color=colors[idx], linestyle='--', marker=markers[idx], markersize=3,
                 label=f'test 1 ({label_suffix})')
        plt.plot(epochs, f_test_2, color=colors[idx], linestyle='--', marker=markers[idx], markersize=3,
                 label=f'test 2 ({label_suffix})')

    # plt.ylim(0.2, 1) # for recall
    # plt.ylim(0, 0.95) # F1
    plt.ylim(0, 0.95) # precision
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metrics, fontsize=12)
    plt.title(f'{metrics} by Epoch for bit = {bit}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{metrics}_scores_plot_quan_{bit}_seeds.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and plot precision scores from training log files.")
    parser.add_argument("--bit", type=int, default=2, help="bit = 2/4/6/8")
    # parser.add_argument("--log_files", nargs=3, required=False,
                        # help="Paths to three training log files (different seeds).")
    parser.add_argument("--metrics", type=str, default="Precision", help="precision/f1/recall score")
    args = parser.parse_args()
    log_files = [f"result_quan_{args.bit}_12345.txt", f"result_quan_{args.bit}_23456.txt", f"result_quan_{args.bit}_34567.txt"]
    plot_f1_scores_multiple(log_files, args.metrics, args.bit)
