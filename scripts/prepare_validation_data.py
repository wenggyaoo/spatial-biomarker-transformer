import os
import shutil
import random
from pathlib import Path


def split_data_for_validation(data_path, val_split=0.2, seed=42):
    """
    Split existing training data to create validation set

    Args:
        data_path: Path to your data directory
        val_split: Fraction of data to use for validation (0.2 = 20%)
        seed: Random seed for reproducible splits
    """

    random.seed(seed)

    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')

    if not os.path.exists(train_path):
        print(f"❌ Training data not found at: {train_path}")
        return

    # Create validation directory
    os.makedirs(val_path, exist_ok=True)

    print(f"Splitting data with {val_split * 100}% for validation...")

    # Process each study
    for study_dir in os.listdir(train_path):
        study_train_path = os.path.join(train_path, study_dir)

        if not os.path.isdir(study_train_path):
            continue

        print(f"Processing study: {study_dir}")

        # Get all regions in this study
        regions = [d for d in os.listdir(study_train_path)
                   if os.path.isdir(os.path.join(study_train_path, d))]

        # Calculate number of validation regions
        num_val_regions = max(1, int(len(regions) * val_split))

        # Randomly select validation regions
        val_regions = random.sample(regions, num_val_regions)

        print(f"  Total regions: {len(regions)}")
        print(f"  Moving {num_val_regions} regions to validation")

        # Create study directory in validation
        study_val_path = os.path.join(val_path, study_dir)
        os.makedirs(study_val_path, exist_ok=True)

        # Move selected regions to validation
        for region in val_regions:
            src_region_path = os.path.join(study_train_path, region)
            dst_region_path = os.path.join(study_val_path, region)

            shutil.move(src_region_path, dst_region_path)
            print(f"    Moved {region} to validation")

    print("✓ Data split completed!")

    # Print summary
    print_data_summary(data_path)


def print_data_summary(data_path):
    """Print summary of train/val split"""

    print("\n=== Data Summary ===")

    for split in ['train', 'val']:
        split_path = os.path.join(data_path, split)
        if os.path.exists(split_path):
            total_regions = 0
            studies = []

            for study_dir in os.listdir(split_path):
                study_path = os.path.join(split_path, study_dir)
                if os.path.isdir(study_path):
                    regions = [d for d in os.listdir(study_path)
                               if os.path.isdir(os.path.join(study_path, d))]
                    total_regions += len(regions)
                    studies.append(f"{study_dir}: {len(regions)} regions")

            print(f"{split.upper()} SET: {total_regions} total regions")
            for study_info in studies:
                print(f"  {study_info}")


if __name__ == "__main__":
    data_path = r"/autofs/bal14/khguo/data_celltype"  # Change this to your data path
    split_data_for_validation(data_path, val_split=0.2)