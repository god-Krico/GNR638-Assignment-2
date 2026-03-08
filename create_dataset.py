import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

# Updated to the dataset required for GNR638
DATASET_DIR = "AID"          # Ensure you extract the downloaded AID dataset here
OUTPUT_DIR = "dataset_splits"

def create_splits(seed):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Creating directories for the full train, few-shot subsets, and validation
    splits = ["train_100", "train_20", "train_05", "val"]
    
    # Safely get only directories (ignoring hidden files/zips)
    classes = [c for c in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, c))]
    
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    for cls in classes:
        class_dir = os.path.join(DATASET_DIR, cls)
        # Assuming AID images are jpg based on assignment PDF
        images = [img for img in os.listdir(class_dir) if img.endswith('.jpg')]

        # Standard split: 80% train (which forms our 100% training baseline), 20% validation
        train_100_imgs, val_imgs = train_test_split(images, test_size=0.20, random_state=seed)
        
        # Few-shot splits: 20% and 5% of the training baseline
        train_20_imgs, _ = train_test_split(train_100_imgs, train_size=0.20, random_state=seed)
        train_05_imgs, _ = train_test_split(train_100_imgs, train_size=0.05, random_state=seed)

        # Copy files to respective directories
        for img in train_100_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(OUTPUT_DIR, "train_100", cls))
            
        for img in train_20_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(OUTPUT_DIR, "train_20", cls))
            
        for img in train_05_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(OUTPUT_DIR, "train_05", cls))

        for img in val_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(OUTPUT_DIR, "val", cls))

    print(f"Dataset splits (Full, 20%, 5%, Val) complete using seed {seed}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create AID dataset splits for GNR638")
    parser.add_argument("--seed", type=int, default=42, help="Assigned random seed for few-shot subsets")
    args = parser.parse_args()
    
    create_splits(args.seed)