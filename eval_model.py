"""
Evaluate a trained model on the validation set for GNR638.

Outputs (in checkpoints/<run_name>/eval_results/):
 - confmat_<run_name>.png
 - confmat_<run_name>_norm.png
 - tsne_embeddings_<run_name>.png
 - summary_eval.txt
 
Requires:
    pip install scikit-learn
"""

import torch
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import itertools

from models import create_model
from datasets import get_dataloader

def load_checkpoint(model, ckpt_path, device):
    ck = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state_dict = ck["model_state_dict"]
    else:
        state_dict = ck
    
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state, strict=False)
    return model

def get_embeddings_and_preds(model, dataloader, device):
    """
    Evaluates the model, collects predictions, and uses a forward hook 
    to extract the pre-classification feature embeddings.
    """
    model.eval()
    preds, labels_all, embeddings = [], [], []
    
    # Dictionary to hold the extracted features from the hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # The input to the final classifier is our embedding
            activation[name] = input[0].detach()
        return hook

    # Register hook to the final classifier layer
    head_names = ("fc", "classifier", "head")
    hook_handle = None
    for hn in head_names:
        if hasattr(model, hn):
            hook_handle = getattr(model, hn).register_forward_hook(get_activation('features'))
            break
            
    if hook_handle is None:
        try:
            cls = model.get_classifier()
            if cls is not None:
                hook_handle = cls.register_forward_hook(get_activation('features'))
        except Exception:
            pass

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            out = model(imgs)
            p = out.argmax(dim=1)
            
            preds.extend(p.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            
            if 'features' in activation:
                # Flatten the features in case they are spatial (B, C, H, W) -> (B, C*H*W) or (B, C)
                emb = activation['features'].view(activation['features'].size(0), -1)
                embeddings.append(emb.cpu().numpy())
                
    if hook_handle:
        hook_handle.remove()
        
    acc = accuracy_score(labels_all, preds)
    cm = confusion_matrix(labels_all, preds)
    
    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
    else:
        embeddings = None
        print("Warning: Could not extract embeddings. Hook failed to attach.")
        
    return acc, cm, np.array(labels_all), embeddings


def plot_confusion(cm, classes, outpath, normalize=False, title="Confusion Matrix"):
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    # Increased figsize specifically for 30 classes
    fig_size = max(10, len(classes) * 0.4)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=90, ha="center", fontsize=8)
    plt.yticks(ticks, classes, fontsize=8)

    # Only annotate text if the grid isn't overwhelmingly large, or use tiny font
    font_size = 6 if len(classes) > 20 else 10
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j]
        if val > 0: # Skip zero values to reduce clutter
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j, i, text, ha="center", va="center", 
                     color="white" if val > thresh else "black", fontsize=font_size)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_embeddings(embeddings, labels, class_names, outpath, method="tsne", title="Feature Embeddings"):
    print(f"Reducing dimensions using {method.upper()}... (this may take a minute)")
    
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
        
    reduced_emb = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    # Turbo colormap is excellent for large discrete classes > 20
    scatter = plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels, cmap='turbo', alpha=0.7, s=15)
    
    # Create a legend
    cbar = plt.colorbar(scatter, ticks=range(len(class_names)))
    cbar.ax.set_yticklabels(class_names)
    
    plt.title(title)
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Using device:", device)

    # Locate the specific checkpoint
    run_dir = Path(args.save_dir) / args.run_name
    ckpt_path = run_dir / f"best_{args.run_name}.pth"
    
    if not ckpt_path.exists():
        print(f"Error: Could not find checkpoint at {ckpt_path}")
        return

    out_dir = run_dir / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_loader, test_ds = get_dataloader(
        root_dir=args.data,
        split="val", # Evaluating on validation since test is hidden
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )
    class_names = test_ds.classes

    print(f"\nEvaluating {args.run_name} -> {ckpt_path}")

    # Build model (we don't need fine_tune_strategy here since we load all weights anyway)
    model = create_model(
        name=args.model,
        num_classes=len(class_names),
        pretrained=False,
        in_chans=3
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device)

    # Evaluate and extract
    acc, cm, labels_all, embeddings = get_embeddings_and_preds(model, test_loader, device)
    print(f"Validation Accuracy: {acc:.4f}")

    # Save confusion matrices
    print("Generating Confusion Matrices...")
    plot_confusion(cm, class_names,
                   out_dir / f"confmat_{args.run_name}.png",
                   normalize=False,
                   title=f"Confusion Matrix: {args.run_name}")

    plot_confusion(cm, class_names,
                   out_dir / f"confmat_{args.run_name}_norm.png",
                   normalize=True,
                   title=f"Normalized Confusion Matrix: {args.run_name}")
                   
    # Save Embedding Visualizations
    if embeddings is not None and args.plot_embeddings:
        plot_embeddings(embeddings, labels_all, class_names, 
                        out_dir / f"{args.embed_method}_{args.run_name}.png", 
                        method=args.embed_method, 
                        title=f"{args.embed_method.upper()} Feature Embeddings: {args.run_name}")

    # Save summary
    with open(out_dir / "summary_eval.txt", "w") as f:
        f.write(f"Run: {args.run_name}\n")
        f.write(f"Validation Accuracy: {acc:.4f}\n")

    print("\nEvaluation complete.")
    print("Results saved in:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="Name of the training run (e.g., resnet50_linear_probe_train_100)")
    parser.add_argument("--model", type=str, required=True, help="Model architecture (e.g., resnet50)")
    parser.add_argument("--data", type=str, default="dataset_splits")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--plot_embeddings", action="store_true", help="Generate PCA/t-SNE plots")
    parser.add_argument("--embed_method", type=str, default="tsne", choices=["tsne", "pca"], help="Method for embedding visualization")
    
    args = parser.parse_args()
    main(args)