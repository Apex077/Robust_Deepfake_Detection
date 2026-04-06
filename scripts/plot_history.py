"""
plot_history.py
---------------
Plots training loss and validation AUC over epochs from the training history CSV.
"""
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="checkpoints/history.csv", help="Path to history.csv")
    p.add_argument("--output_dir", default="results/overfitting/", help="Output directory")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    history_path = Path(args.history)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        print(f"Error: {history_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(history_path)

    epochs = df["epoch"]
    train_loss = df["train_loss"]
    val_auc = df["val_auc"]

    # 1. Plot Training Loss and Validation AUC
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color)
    ax1.plot(epochs, train_loss, color=color, marker="o", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("Validation AUC", color=color)
    ax2.plot(epochs, val_auc, color=color, marker="s", label="Val AUC")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Training History")
    
    out_file = out_dir / "training_history.png"
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"Plot saved to {out_file}")
