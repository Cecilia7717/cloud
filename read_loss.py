import re
import matplotlib.pyplot as plt
import numpy as np

def parse_f1_scores(log_file):
    """Extract F1 scores from training logs, handling missing epochs"""
    epochs = []
    f1_1 = []
    f_test_1 = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check for epoch line
            epoch_match = re.search(r"Epoch\[(\d+)\]", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                if current_epoch >= 100:
                    break
                test = False
                # Look for F1 in the next few lines
                for _ in range(5):  # Check next 5 lines for metrics
                    next_line = f.readline()
                    if not next_line:
                        break
                    if "Test" in next_line:
                        test = True
                    if "Loss" in next_line:
                        if test:
                            matches = re.findall(r"\d+\.\d+", next_line)
                            if len(matches) >= 1:
                                epochs.append(current_epoch)
                                f_test_1.append(float(matches[0]))
                            break
                        else:
                            matches = re.findall(r"\d+\.\d+", next_line)
                            if len(matches) >= 1:
                                f1_1.append(float(matches[0]))
                            break
    return epochs, f1_1, f_test_1

def plot_f1_scores(epochs, f1_1, f_test_1):
    """Create a line plot of F1 scores over epochs"""
    plt.figure(figsize=(10, 6))
    
    # Plot both F1 scores
    line1, = plt.plot(epochs, f1_1, 'g-', linewidth=2, marker='o', markersize=3, label='train loss')
    line3, = plt.plot(epochs, f_test_1, 'g--', linewidth=2, marker='o', markersize=3, label='test loss')
    plt.ylim(0.4, 0.8)
    # Customize plot
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.title('loss by Epoch for bit = 8', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Adjust x-axis to show actual epoch numbers
    plt.xticks(epochs)
    
    # Rotate x-labels if many epochs
    if len(epochs) > 10:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('loss_plot_quan_8.png', dpi=300, bbox_inches='tight') # place to change the name of the output file
    plt.show()

if __name__ == "__main__":
    log_file = "result_quan_8.txt"  # Change to your file path
    epochs, f1_c1, f_test_1 = parse_f1_scores(log_file)
    
    if epochs:
        print(f"Found data for epochs: {epochs}")
        print(f"Class 1 precision: {f1_c1}")
        print(f"Class 2 precision: {f_test_1}")
        
        plot_f1_scores(epochs, f1_c1, f_test_1)
    else:
        print("No valid F1 data found in the log file!")