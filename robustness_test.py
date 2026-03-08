"""
Robustness test suite for GNR638 Assignment 2 (Scenario 4.4).

Usage:
    python robustness_test.py --run_names resnet50_full_train_100 densenet121_full_train_100 --batch_size 32

Outputs (in checkpoints/robustness_results/):
 - accuracies_<corruption>.csv    
 - metrics_summary.txt (Contains Corruption Error and Relative Robustness)
 - accuracies_<corruption>.png    
"""

import argparse
from pathlib import Path
import math

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

from models import create_model
from datasets import get_dataloader

CHECKPOINT_DIR = Path("checkpoints")
OUT_DIR = CHECKPOINT_DIR / "robustness_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def evaluate_model_on_loader(model, loader, device):
    model.eval()
    preds_all = []
    labels_all = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)
        preds = out.argmax(dim=1)
        preds_all.extend(preds.cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())
    
    total = len(labels_all)
    acc = 0.0
    if total > 0:
        acc = (np.array(preds_all) == np.array(labels_all)).sum() / total
    return acc

def load_checkpoint_into_model(model, ckpt_path, device):
    ck = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    else:
        sd = ck
        
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_names", nargs='+', required=True, help="List of run names to evaluate (e.g., resnet50_full_train_100)")
    parser.add_argument("--data", type=str, default="dataset_splits", help="root dataset folder")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Using device:", device)

    # 1. Verify Checkpoints
    ckpt_map = {}
    for run in args.run_names:
        # Assuming the model name is the first part of the run_name (e.g., resnet50_...)
        model_arch = run.split("_")[0] 
        ckpt = CHECKPOINT_DIR / run / f"best_{run}.pth"
        if ckpt.exists():
            ckpt_map[run] = {"arch": model_arch, "path": ckpt}
        else:
            print(f"WARNING: Checkpoint missing for {run} at {ckpt}; skipping.")

    if not ckpt_map:
        print("No valid checkpoints found. Exiting.")
        return

    # 2. Define Corruptions for Scenario 4.4
    # The first value in each list is the "Clean" baseline
    corruptions = {
        "gaussian_noise": [0.0, 0.05, 0.10, 0.20], 
        "motion_blur":    [0.0, 1.0, 2.0, 3.0],  # 0 is clean, 1/2/3 are blur radii
        "brightness":     [1.0, 0.5, 1.5, 2.0]   # 1.0 is clean
    }

    results_acc = {c: {run: [] for run in ckpt_map.keys()} for c in corruptions.keys()}

    # 3. Evaluate
    for corr_name, severities in corruptions.items():
        print(f"\n=== Evaluating {corr_name} ===")
        
        for sev in severities:
            # Determine if this is the "clean" run
            is_clean = (corr_name == "brightness" and sev == 1.0) or (corr_name != "brightness" and sev == 0.0)
            
            # Use our unified get_dataloader to handle the transforms
            c_type = None if is_clean else corr_name
            loader, ds = get_dataloader(
                root_dir=args.data, split="val", batch_size=args.batch_size, 
                img_size=args.img_size, shuffle=False, num_workers=4,
                pin_memory=(device.type=="cuda"), corruption_type=c_type, severity=sev
            )
            
            print(f"  Severity: {sev}")
            for run_name, info in ckpt_map.items():
                model = create_model(name=info["arch"], num_classes=len(ds.classes), pretrained=False, in_chans=3)
                model = load_checkpoint_into_model(model, info["path"], device).to(device)
                
                acc = evaluate_model_on_loader(model, loader, device)
                results_acc[corr_name][run_name].append(acc)
                print(f"    {run_name}: {acc:.4f}")

    # 4. Calculate Metrics and Save Outputs
    summary_path = OUT_DIR / "metrics_summary.txt"
    with open(summary_path, "w") as f_sum:
        f_sum.write("Scenario 4.4: Robustness Metrics\n")
        f_sum.write("="*40 + "\n")

        for corr_name, severities in corruptions.items():
            f_sum.write(f"\n--- Corruption: {corr_name.upper()} ---\n")
            
            # Save raw accuracy CSV
            csv_path = OUT_DIR / f"accuracies_{corr_name}.csv"
            with open(csv_path, "w", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["severity"] + list(ckpt_map.keys()))
                
                for i, sev in enumerate(severities):
                    row = [sev]
                    for run in ckpt_map.keys():
                        row.append(f"{results_acc[corr_name][run][i]:.4f}")
                    writer.writerow(row)
            
            # Log specific GNR638 metrics to summary
            for run in ckpt_map.keys():
                f_sum.write(f"Model Run: {run}\n")
                clean_acc = results_acc[corr_name][run][0] # Index 0 is always the clean baseline
                f_sum.write(f"  Clean Accuracy: {clean_acc:.4f}\n")
                
                for i, sev in enumerate(severities[1:], start=1):
                    corrupted_acc = results_acc[corr_name][run][i]
                    corruption_error = 1.0 - corrupted_acc
                    # Prevent division by zero if a model totally collapses on clean data
                    rel_robustness = corrupted_acc / clean_acc if clean_acc > 0 else 0.0
                    
                    f_sum.write(f"  Severity {sev}:\n")
                    f_sum.write(f"    Acc: {corrupted_acc:.4f}\n")
                    f_sum.write(f"    Corruption Error: {corruption_error:.4f}\n")
                    f_sum.write(f"    Relative Robustness: {rel_robustness:.4f}\n")
                f_sum.write("\n")

            # Plot overlay
            plt.figure(figsize=(8,5))
            for run in ckpt_map.keys():
                plt.plot(severities, results_acc[corr_name][run], marker='o', label=run)
            plt.xlabel("Severity")
            plt.ylabel("Validation Accuracy")
            plt.title(f"Robustness Shift: {corr_name}")
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"plot_{corr_name}.png")
            plt.close()

    print(f"\nEvaluation complete. Reports and plots saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()