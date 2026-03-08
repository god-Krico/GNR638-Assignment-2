"""
Robust Data pipeline for AID dataset.
"""

import os
import math
from PIL import Image, UnidentifiedImageError, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from collections import Counter
from typing import Tuple, List

# --- Custom Corruption Transforms for Scenario 4.4 ---

class AddGaussianNoise(object):
    def __init__(self, sigma=0.05):
        self.sigma = sigma
        
    def __call__(self, tensor):
        # Add noise and clamp values to valid [0, 1] range before normalization
        noise = torch.randn_like(tensor) * self.sigma
        return torch.clamp(tensor + noise, 0.0, 1.0)

class MotionBlur(object):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        # Using PIL's BoxBlur as a basic proxy for motion blur
        # For strict directional motion blur, cv2.filter2D would be needed, 
        # but PIL is safer for avoiding cross-platform installation issues.
        return img.filter(ImageFilter.BoxBlur(self.radius))

class BrightnessShift(object):
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        return TF.adjust_brightness(img, self.factor)

# --- Main Dataset Class ---

class AIDDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train_100", img_size: int = 224, transform=None, skip_invalid: bool = True):
        self.split = split
        self.root = os.path.join(root_dir, split)
        if not os.path.isdir(self.root):
            raise ValueError(f"Split directory does not exist: {self.root}")

        self.classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        self.invalid_files = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    full = os.path.join(cls_dir, fname)
                    if skip_invalid:
                        try:
                            with Image.open(full) as im:
                                im.verify() 
                            self.samples.append((full, self.class_to_idx[cls]))
                        except (UnidentifiedImageError, OSError, ValueError) as e:
                            self.invalid_files.append((full, str(e)))
                    else:
                        self.samples.append((full, self.class_to_idx[cls]))

        self.transform = transform or self.default_transform(img_size)

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid image samples found for split '{split}' in {self.root}")

    def default_transform(self, img_size: int):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Failed to open image {path} : {e}")
        
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_invalid_files(self) -> List[Tuple[str, str]]:
        return self.invalid_files


def get_transforms(img_size: int = 224, split: str = "train_100", randaugment: bool = False, corruption_type: str = None, severity: float = None):
    """Return torchvision transforms for a given split, with optional evaluation corruptions."""
    
    # Base transforms for PIL Images
    t_base = []
    
    # If evaluating with corruptions that apply to PIL images
    if split == "val" and corruption_type == "motion_blur":
        t_base.append(MotionBlur(radius=int(severity) if severity else 2))
    elif split == "val" and corruption_type == "brightness":
        t_base.append(BrightnessShift(factor=severity if severity else 1.5))

    # Any train split gets augmentations
    if split.startswith("train"):
        t_base.extend([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)
        ])
        if randaugment:
            try:
                from torchvision.transforms import RandAugment
                t_base.append(RandAugment())
            except Exception:
                pass
    else:
        # Validation clean base
        t_base.append(transforms.Resize((img_size, img_size)))

    # Convert to Tensor
    t_base.append(transforms.ToTensor())

    # Add Tensor-based corruptions BEFORE normalization
    if split == "val" and corruption_type == "gaussian_noise":
        # Using default sigma=0.05 if not provided
        t_base.append(AddGaussianNoise(sigma=severity if severity else 0.05))

    # Standard ImageNet Normalization
    t_base.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(t_base)


def get_dataloader(root_dir: str = "dataset_splits",
                   split: str = "train_100",
                   batch_size: int = 32,
                   img_size: int = 224,
                   shuffle: bool = True,
                   num_workers: int = 4, # Increased default for faster loading
                   pin_memory: bool = True,
                   randaugment: bool = False,
                   corruption_type: str = None,
                   severity: float = None) -> Tuple[DataLoader, AIDDataset]:
    
    tf = get_transforms(img_size=img_size, split=split, randaugment=randaugment, 
                        corruption_type=corruption_type, severity=severity)
    ds = AIDDataset(root_dir=root_dir, split=split, img_size=img_size, transform=tf, skip_invalid=True)
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=(split.startswith("train")) and shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory)
    return dl, ds


def dataset_stats(root_dir: str = "dataset_splits"):
    stats = {}
    splits = ("train_100", "train_20", "train_05", "val")
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            stats[split] = None
            continue
        class_counts = {}
        total = 0
        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            cnt = len([f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
            class_counts[cls] = cnt
            total += cnt
        stats[split] = {"total": total, "per_class": class_counts}
        
    for split, info in stats.items():
        if info is None:
            print(f"{split}: not found")
        else:
            print(f"{split}: {info['total']} images")
    return stats

def find_corrupted(root_dir: str = "dataset_splits") -> List[Tuple[str, str]]:
    corrupted = []
    for split in ("train_100", "train_20", "train_05", "val"):
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    continue
                path = os.path.join(cls_dir, fname)
                try:
                    with Image.open(path) as im:
                        im.verify()
                except Exception as e:
                    corrupted.append((path, str(e)))
    if len(corrupted) > 0:
        print(f"Found {len(corrupted)} corrupted images.")
    else:
        print("No corrupted images found.")
    return corrupted