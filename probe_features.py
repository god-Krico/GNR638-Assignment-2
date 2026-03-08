"""
Layer-Wise Feature Probing for GNR638 Assignment 2 (Scenario 4.5).

Usage:
    python probe_features.py --model resnet50 --data dataset_splits

Outputs (in checkpoints/probing_results/<model>/):
 - acc_vs_depth.png       (Validation accuracy vs. network depth)
 - feature_norms.txt      (L2 norm statistics across selected layers)
 - pca_depth_0_early.png  (PCA on early layers)
 - pca_depth_1_middle.png (PCA on middle layers)
 - pca_depth_2_final.png  (PCA on final layers)
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models import create_model
from datasets import get_dataloader, AIDDataset, get_transforms

OUT_ROOT = Path("checkpoints/probing_results")

def get_balanced_subset_indices(dataset, samples_per_class=30):
    """
    Extracts indices for a fixed subset (30 classes, 30 samples/class).
    Requirement: Scenario 4.5 PCA subset.
    """
    class_counts = {i: 0 for i in range(len(dataset.classes))}
    subset_indices = []
    
    for idx in range(len(dataset)):
        _, label = dataset.samples[idx]
        if class_counts[label] < samples_per_class:
            subset_indices.append(idx)
            class_counts[label] += 1
            
        if len(subset_indices) == len(dataset.classes) * samples_per_class:
            break
            
    return subset_indices

@torch.no_grad()
def extract_and_cache_features(model, dataloader, device):
    """
    Passes data through the backbone and caches the Global Average Pooled features 
    for the early, middle, and final layers to save compute time.
    """
    model.eval()
    
    cached_features = {0: [], 1: [], 2: []} # depths: 0 (early), 1 (mid), 2 (final)
    cached_labels = []
    
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        cached_labels.extend(labels.numpy())
        
        # model(...) returns a list of tensors when features_only=True
        feature_maps = model(imgs) 
        
        for depth_idx, fmap in enumerate(feature_maps):
            # Apply Global Average Pooling (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            pooled = fmap.mean(dim=[2, 3])
            cached_features[depth_idx].append(pooled.cpu())
            
    # Concatenate all batches
    for d in cached_features.keys():
        cached_features[d] = torch.cat(cached_features[d], dim=0)
        
    return cached_features, torch.tensor(cached_labels)

def train_linear_probe(features_train, labels_train, features_val, labels_val, num_classes=30, device='cpu'):
    """
    Trains a simple linear classifier on the extracted features using LBFGS.
    """
    in_features = features_train.shape[1]
    classifier = nn.Linear(in_features, num_classes).to(device)
    
    features_train = features_train.to(device)
    labels_train = labels_train.to(device)
    features_val = features_val.to(device)
    labels_val = labels_val.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # LBFGS is incredibly fast for strictly linear problems over cached features
    optimizer = torch.optim.LBFGS(classifier.parameters(), max_iter=100)
    
    def closure():
        optimizer.zero_grad()
        output = classifier(features_train)
        loss = criterion(output, labels_train)
        loss.backward()
        return loss
        
    classifier.train()
    optimizer.step(closure)
    
    classifier.eval()
    with torch.no_grad():
        out_val = classifier(features_val)
        preds = out_val.argmax(dim=1)
        acc = (preds == labels_val).float().mean().item()
        
    return acc

def plot_pca_fixed_subset(features, labels, class_names, outpath, title="PCA Embeddings"):
    """Generates the required PCA 2D plot for the layer features."""
    reducer = PCA(n_components=2, random_state=42)
    reduced_emb = reducer.fit_transform(features.numpy())
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels.numpy(), cmap='turbo', alpha=0.8, s=20)
    
    cbar = plt.colorbar(scatter, ticks=range(len(class_names)))
    cbar.ax.set_yticklabels(class_names)
    
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture")
    parser.add_argument("--data", type=str, default="dataset_splits", help="Dataset root")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_ROOT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Scenario 4.5: Layer-Wise Probing for {args.model} ---")

    # 1. Load Model with extract_features=True
    # This automatically grabs out_indices=(0, 2, 4) in our models.py dispatcher
    model = create_model(name=args.model, extract_features=True, pretrained=True).to(device)
    
    # 2. Setup Data
    train_loader, train_ds = get_dataloader(root_dir=args.data, split="train_100", batch_size=args.batch_size, img_size=args.img_size, shuffle=False)
    val_loader, val_ds = get_dataloader(root_dir=args.data, split="val", batch_size=args.batch_size, img_size=args.img_size, shuffle=False)
    
    # 3. Cache Features (Passes through CNN once)
    print("Extracting features from training set...")
    train_feats, train_labels = extract_and_cache_features(model, train_loader, device)
    
    print("Extracting features from validation set...")
    val_feats, val_labels = extract_and_cache_features(model, val_loader, device)
    
    # 4. Train Probes and Calculate Norms
    depth_names = {0: "Early", 1: "Middle", 2: "Final"}
    accuracies = []
    
    norm_log_path = out_dir / "feature_norms.txt"
    with open(norm_log_path, "w") as f_norm:
        f_norm.write(f"Feature Norm Statistics for {args.model}\n")
        f_norm.write("=========================================\n")
        
        for depth in [0, 1, 2]:
            print(f"\nTraining Linear Probe for {depth_names[depth]} Layer (Depth {depth})...")
            
            # Train Probe
            acc = train_linear_probe(train_feats[depth], train_labels, val_feats[depth], val_labels, num_classes=len(train_ds.classes), device=device)
            accuracies.append(acc)
            print(f"  -> Validation Accuracy: {acc:.4f}")
            
            # Calculate L2 Norm Statistics
            norms = torch.norm(val_feats[depth], p=2, dim=1)
            mean_norm = norms.mean().item()
            std_norm = norms.std().item()
            
            f_norm.write(f"Depth {depth} ({depth_names[depth]}):\n")
            f_norm.write(f"  Feature Dimension: {val_feats[depth].shape[1]}\n")
            f_norm.write(f"  Mean L2 Norm: {mean_norm:.4f}\n")
            f_norm.write(f"  Std L2 Norm:  {std_norm:.4f}\n\n")

    # 5. Plot Accuracy vs Depth
    plt.figure(figsize=(7, 5))
    plt.plot([0, 1, 2], accuracies, marker='o', linestyle='-', color='indigo', markersize=8)
    plt.xticks([0, 1, 2], [depth_names[0], depth_names[1], depth_names[2]])
    plt.xlabel("Network Depth")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Semantic Abstraction: Accuracy vs. Depth ({args.model})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "acc_vs_depth.png")
    plt.close()

    # 6. Generate PCA plots on the strict subset (30 samples/class)
    print("\nGenerating PCA plots on fixed subset...")
    subset_indices = get_balanced_subset_indices(val_ds, samples_per_class=30)
    subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
    
    # Create a special loader just for the PCA subset to guarantee order
    tf = get_transforms(img_size=args.img_size, split="val")
    pca_ds = AIDDataset(root_dir=args.data, split="val", img_size=args.img_size, transform=tf)
    pca_loader = DataLoader(pca_ds, batch_size=args.batch_size, sampler=subset_sampler)
    
    pca_feats, pca_labels = extract_and_cache_features(model, pca_loader, device)
    
    for depth in [0, 1, 2]:
        plot_pca_fixed_subset(
            features=pca_feats[depth], 
            labels=pca_labels, 
            class_names=val_ds.classes, 
            outpath=out_dir / f"pca_depth_{depth}_{depth_names[depth].lower()}.png",
            title=f"PCA Embeddings - {depth_names[depth]} Layer ({args.model})"
        )

    print(f"\nDone! Results saved to {out_dir}")

if __name__ == "__main__":
    main()