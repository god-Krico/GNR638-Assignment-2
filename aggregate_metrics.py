"""
Post-training aggregation script for GNR638.
Calculates Few-Shot performance drops and plots Fine-tuning Strategy accuracies.
"""

import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints")

def calculate_few_shot_drop(model_name):
    print(f"\n--- Few-Shot Drop Analysis for {model_name} ---")
    splits = {"100%": "train_100", "20%": "train_20", "5%": "train_05"}
    accuracies = {}

    for name, split in splits.items():
        run_name = f"{model_name}_full_{split}"
        ckpt_path = CHECKPOINT_DIR / run_name / f"best_{run_name}.pth"
        
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            accuracies[name] = ckpt["val_acc"]
        else:
            print(f"Warning: Checkpoint not found for {run_name}")
            accuracies[name] = None

    if accuracies.get("100%") and accuracies.get("5%"):
        acc_100 = accuracies["100%"]
        acc_5 = accuracies["5%"]
        relative_drop = (acc_100 - acc_5) / acc_100
        
        print(f"100% Data Validation Acc: {acc_100:.4f}")
        print(f" 20% Data Validation Acc: {accuracies['20%']:.4f}")
        print(f"  5% Data Validation Acc: {acc_5:.4f}")
        print(f"Relative Performance Drop (100% vs 5%): {relative_drop:.4f} ({relative_drop*100:.2f}%)")
        return relative_drop
    return None

def plot_finetune_strategies(model_name):
    print(f"\n--- Strategy Comparison Plot for {model_name} ---")
    
    # We map strategies to an approximate percentage of unfrozen parameters based on the assignment rules
    strategies = {
        "linear_probe": 1.0,      # roughly 1% params (classifier only)
        "selective_20": 20.0,     # exactly 20%
        "last_block": 40.0,       # roughly 40% (depends on architecture, approximated for plot)
        "full": 100.0             # 100% params
    }
    
    x_percentages = []
    y_accuracies = []
    labels = []

    for strategy, pct in strategies.items():
        run_name = f"{model_name}_{strategy}_train_100"
        ckpt_path = CHECKPOINT_DIR / run_name / f"best_{run_name}.pth"
        
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            x_percentages.append(pct)
            y_accuracies.append(ckpt["val_acc"])
            labels.append(strategy)
        else:
            print(f"Warning: Checkpoint not found for {run_name}")

    if len(x_percentages) > 1:
        plt.figure(figsize=(8, 5))
        plt.plot(x_percentages, y_accuracies, marker='s', linestyle='-', color='b')
        
        for i, label in enumerate(labels):
            plt.annotate(label, (x_percentages[i], y_accuracies[i]), textcoords="offset points", xytext=(0,10), ha='center')

        plt.title(f"Validation Accuracy vs Unfrozen Parameters ({model_name})")
        plt.xlabel("Approximate Percentage of Unfrozen Parameters (%)")
        plt.ylabel("Validation Accuracy")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylim(0, 1.0)
        
        out_path = CHECKPOINT_DIR / f"{model_name}_strategy_comparison.png"
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    # Ensure you run this after all training steps are complete
    models = ["resnet50", "densenet121", "efficientnet_b0"]
    for model in models:
        calculate_few_shot_drop(model)
        plot_finetune_strategies(model)