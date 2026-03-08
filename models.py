"""
Organized model factory for GNR638 Assignment 2.

Provides:
 - create_model(name, ...)  # unified dispatcher handling all fine-tuning strategies
 - count_parameters(model, verbose=False)
 - calculate_macs_flops(model, input_size)

Requires:
    pip install timm thop
"""

from typing import Tuple
import timm
import torch
import torch.nn as nn

try:
    from thop import profile
except ImportError:
    profile = None


def _unfreeze_classifier_head(model):
    """Unfreeze only the final classification layer."""
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False
        
    head_names = ("fc", "classifier", "head")
    unfroze = False

    for hn in head_names:
        if hasattr(model, hn):
            head = getattr(model, hn)
            try:
                for p in head.parameters():
                    p.requires_grad = True
                unfroze = True
            except Exception:
                pass

    try:
        cls = model.get_classifier()
        if cls is not None:
            for p in cls.parameters():
                p.requires_grad = True
            unfroze = True
    except Exception:
        pass

    return unfroze


def _unfreeze_last_block(model, model_name: str):
    """
    Unfreezes the classifier and the last major block of the CNN.
    Architecture specific logic is required due to different naming conventions.
    """
    _unfreeze_classifier_head(model)
    
    n = model_name.lower()
    block_unfrozen = False

    # Identify the last block based on architecture family
    if "resnet" in n:
        # ResNets usually have layer1, layer2, layer3, layer4
        if hasattr(model, 'layer4'):
            for p in model.layer4.parameters():
                p.requires_grad = True
            block_unfrozen = True
            
    elif "densenet" in n:
        # DenseNets have features.denseblock4
        if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
            for p in model.features.denseblock4.parameters():
                p.requires_grad = True
            block_unfrozen = True
            
    elif "efficientnet" in n:
        # EfficientNets in timm have blocks, we unfreeze the last one
        if hasattr(model, 'blocks'):
            for p in model.blocks[-1].parameters():
                p.requires_grad = True
            block_unfrozen = True
            
    elif "convnext" in n:
        # ConvNeXt has stages, unfreeze the last stage
        if hasattr(model, 'stages'):
            for p in model.stages[-1].parameters():
                p.requires_grad = True
            block_unfrozen = True
            
    elif "inception" in n:
        # Inception v3 has Mixed_7c as the last major block
        if hasattr(model, 'Mixed_7c'):
            for p in model.Mixed_7c.parameters():
                p.requires_grad = True
            block_unfrozen = True

    if not block_unfrozen:
        print(f"Warning: Could not automatically identify the last block for {model_name}. Only head is unfrozen.")


def _unfreeze_percentage(model, percentage: float = 0.20):
    """
    Unfreezes exactly `percentage` of the total parameters, starting from the 
    top (classifier) and moving backwards through the network. This represents
    adapting the highest-level semantic features first.
    """
    total_params = sum(p.numel() for p in model.parameters())
    target_unfrozen = percentage * total_params
    
    # Freeze all first
    for p in model.parameters():
        p.requires_grad = False
        
    current_unfrozen = 0
    # Iterate backwards through the parameters
    for name, param in reversed(list(model.named_parameters())):
        param.requires_grad = True
        current_unfrozen += param.numel()
        if current_unfrozen >= target_unfrozen:
            break
            
    return current_unfrozen, total_params


def create_model(name: str,
                 num_classes: int = 30, # Default changed to 30 for AID Dataset
                 pretrained: bool = True,
                 in_chans: int = 3,
                 fine_tune_strategy: str = "full",
                 extract_features: bool = False) -> nn.Module:
    """
    Unified dispatcher that creates a model and applies the required assignment strategy.

    Args:
        name: timm model string (e.g., "resnet50", "densenet121", "efficientnet_b0", "convnext_tiny").
        num_classes: number of output classes (30 for AID).
        fine_tune_strategy: "full", "linear_probe", "last_block", or "selective_20".
        extract_features: If True, strips the classifier and returns intermediate layer outputs.
    """
    
    # Feature extraction setup for Scenario 4.5
    if extract_features:
        # out_indices=(0, 2, 4) generally maps to early, middle, and final layer blocks in timm
        model = timm.create_model(name, pretrained=pretrained, in_chans=in_chans, 
                                  features_only=True, out_indices=(0, 2, 4))
        # Freeze backbone for probing
        for p in model.parameters():
            p.requires_grad = False
        return model

    # Standard classification setup
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    # Apply fine-tuning strategies for Scenario 4.2
    if fine_tune_strategy == "full":
        pass # Default timm behavior is full requires_grad=True
    
    elif fine_tune_strategy == "linear_probe":
        _unfreeze_classifier_head(model)
        
    elif fine_tune_strategy == "last_block":
        _unfreeze_last_block(model, name)
        
    elif fine_tune_strategy == "selective_20":
        unfrozen, total = _unfreeze_percentage(model, 0.20)
        print(f"Selective 20% Unfreeze: Unfrozen {unfrozen} / {total} params ({100*unfrozen/total:.2f}%)")
        
    else:
        raise ValueError(f"Unknown fine_tune_strategy: {fine_tune_strategy}")

    return model


def count_parameters(model: nn.Module, verbose: bool = False) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,} ({100.0 * trainable / total:.2f}%)")
    return total, trainable


def calculate_macs_flops(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
    """
    Calculates MACs and FLOPs using the `thop` library.
    Ensure thop is installed: pip install thop
    """
    if profile is None:
        print("Please install thop to calculate MACs and FLOPs: pip install thop")
        return None, None
        
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2 # 1 MAC is approximately 2 FLOPs (one multiply, one add)
    
    print(f"MACs: {macs / 1e9:.3f} G")
    print(f"FLOPs: {flops / 1e9:.3f} G")
    return macs, flops


# Quick sanity test
if __name__ == "__main__":
    print("Testing Linear Probe on ResNet50...")
    m = create_model("resnet50", num_classes=30, pretrained=False, fine_tune_strategy="linear_probe")
    count_parameters(m, verbose=True)
    calculate_macs_flops(m)
    
    print("\nTesting 20% Selective Unfreeze on DenseNet121...")
    m2 = create_model("densenet121", num_classes=30, pretrained=False, fine_tune_strategy="selective_20")
    count_parameters(m2, verbose=True)