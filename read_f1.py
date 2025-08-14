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
            # Check for epoch line
            epoch_match = re.search(r"Epoch\[(\d+)\]", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                test = False
                # Look for F1 in the next few lines
                for _ in range(5):  # Check next 5 lines for metrics
                    next_line = f.readline()
                    if not next_line:
                        break
                    if "Test" in next_line:
                        test = True
                    if metrics in next_line:
                        if test:
                            matches = re.findall(r"\d+\.\d+", next_line)
                            if len(matches) >= 2:
                                epochs.append(current_epoch)
                                f_test_1.append(float(matches[0]))
                                f_test_2.append(float(matches[1]))
                            break
                        else:
                            matches = re.findall(r"\d+\.\d+", next_line)
                            if len(matches) >= 2:
                                f1_1.append(float(matches[0]))
                                f1_2.append(float(matches[1]))
                            break
    return epochs, f1_1, f1_2, f_test_1, f_test_2

def plot_f1_scores(epochs, f1_1, f1_2, f_test_1, f_test_2, metrics):
    """Create a line plot of F1 scores over epochs"""
    plt.figure(figsize=(10, 6))
    
    # Plot both F1 scores
    line1, = plt.plot(epochs, f1_1, 'g-', linewidth=2, marker='o', markersize=3, label='precision train 1')
    line3, = plt.plot(epochs, f_test_1, 'g--', linewidth=2, marker='o', markersize=3, label='precision test 1')
    line2, = plt.plot(epochs, f1_2, 'r-', linewidth=2, marker='s', markersize=3, label='precision train 2')
    line4, = plt.plot(epochs, f_test_2, 'r--', linewidth=2, marker='s', markersize=3, label='precision test 2')
    
    # Customize plot
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metrics, fontsize=12)
    plt.title('{} by Epoch'.format(metrics), fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Adjust x-axis to show actual epoch numbers
    plt.xticks(epochs)
    
    # Rotate x-labels if many epochs
    if len(epochs) > 10:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('{}_scores_plot_quan.png'.format(metrics), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and plot precision scores from a training log file.")
    parser.add_argument("--log_file", type=str, default="result_quan_1047.txt", help="Path to the training log file.")
    parser.add_argument("--metrics", type=str, default="Precision", help="precision/f1/recall score")
    
    args = parser.parse_args()
    log_file = args.log_file
    metrics = args.metrics
    epochs, f1_c1, f1_c2, f_test_1, f_test_2 = parse_f1_scores(log_file, metrics)
    
    if epochs:
        print(f"Found data for epochs: {epochs}")
        print(f"Class 1 precision: {f1_c1}")
        print(f"Class 2 precision: {f1_c2}")
        
        plot_f1_scores(epochs, f1_c1, f1_c2, f_test_1, f_test_2, metrics)
    else:
        print("No valid F1 data found in the log file!")